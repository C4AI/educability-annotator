"""torchrun coordination through small files on the shared output filesystem.

No process group or tensor collectives are needed for independent inference.
One active job owns the output, including across partial rank failures.
"""

from __future__ import annotations

import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .common import AnnotationError, atomic_json, digest, read_json, sync_directory


@dataclass(frozen=True)
class Topology:
    rank: int
    world_size: int
    local_rank: int
    group_key: str

    @classmethod
    def from_environment(cls):
        values = [os.environ.get(name) for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")]
        if any(value is not None for value in values) and not all(value is not None for value in values):
            raise AnnotationError("RANK, WORLD_SIZE e LOCAL_RANK precisam estar definidos juntos.")
        if all(value is None for value in values):
            return cls(0, 1, 0, uuid.uuid4().hex)
        try:
            rank, world, local = map(int, values)
        except ValueError as exc:
            raise AnnotationError("Ranks inválidos no ambiente.") from exc
        if world < 1 or not 0 <= rank < world or local < 0:
            raise AnnotationError("Topologia distribuída inválida.")
        run_id = os.environ.get("TORCHELASTIC_RUN_ID")
        if not run_id or run_id == "none":
            raise AnnotationError("Use torchrun --standalone ou um --rdzv-id único por execução.")
        key = digest({
            key: os.environ.get(key) for key in (
                "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT", "MASTER_ADDR", "MASTER_PORT",
            )
        })
        return cls(rank, world, local, key)


class Job:
    def __init__(self, output: Path, topology: Topology, timeout: float = 300):
        self.output = output
        self.topology = topology
        self.timeout = timeout
        self.lock = output / ".run-lock"
        self.attempt = None
        self.owner = None

    def _wait(self, predicate, description: str):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.1)
        raise AnnotationError(f"Tempo esgotado aguardando {description}.")

    def join(self, signature: dict, resume: bool, recover_lock: bool = False) -> None:
        rank = self.topology.rank
        if rank == 0:
            if recover_lock:
                if not resume:
                    raise AnnotationError("--recover-lock exige --resume.")
                if self.lock.exists():
                    abandoned = self.output / f"abandoned-lock-{uuid.uuid4().hex}"
                    os.rename(self.lock, abandoned)
                    sync_directory(self.output)
            try:
                self.lock.mkdir()
            except FileExistsError as exc:
                raise AnnotationError(
                    "Saída bloqueada por outra execução. Após parar todos os processos anteriores, "
                    "use --resume --recover-lock."
                ) from exc
            self.owner = {
                "group_key": self.topology.group_key, "attempt_id": uuid.uuid4().hex,
                "world_size": self.topology.world_size, "pid": os.getpid(), "host": socket.gethostname(),
                "resume": resume,
            }
            atomic_json(self.lock / "owner.json", self.owner)
        else:
            def owner_ready():
                owner_file = self.lock / "owner.json"
                if not owner_file.exists():
                    return None
                owner = read_json(owner_file)
                return owner if owner["group_key"] == self.topology.group_key else None
            self.owner = self._wait(owner_ready, "rank 0 adquirir a saída")
        if self.owner["world_size"] != self.topology.world_size or self.owner["resume"] != resume:
            raise AnnotationError("Parâmetros distribuídos divergem entre ranks.")
        self.attempt = self.output / "attempts" / self.owner["attempt_id"]
        atomic_json(self.attempt / f"join-{rank}.json", signature)
        if rank == 0:
            self._wait(
                lambda: all((self.attempt / f"join-{i}.json").exists() for i in range(self.topology.world_size)),
                "todos os ranks ingressarem",
            )
            signatures = [read_json(self.attempt / f"join-{i}.json") for i in range(self.topology.world_size)]
            if any(item != signature for item in signatures):
                raise AnnotationError("Precisão, versões ou parâmetros incompatíveis entre ranks.")

    def publish_ready(self, value: dict) -> None:
        atomic_json(self.attempt / "ready.json", value)

    def ready(self) -> dict:
        def available():
            failure = self.attempt / "error-0.json"
            if failure.exists():
                raise AnnotationError(read_json(failure)["error"])
            path = self.attempt / "ready.json"
            return read_json(path) if path.exists() else None
        return self._wait(available, "preparação da execução")

    def assert_owner(self) -> None:
        owner = read_json(self.lock / "owner.json")
        if owner["attempt_id"] != self.owner["attempt_id"]:
            raise AnnotationError("A execução perdeu a posse da saída.")

    def fail(self, error: BaseException) -> None:
        if self.attempt is not None:
            atomic_json(self.attempt / f"error-{self.topology.rank}.json", {"error": str(error)})
        # Keep the lock on failure: surviving ranks must never race a new job.

    def finish_rank(self) -> None:
        atomic_json(self.attempt / f"done-{self.topology.rank}.json", {"done": True})

    def wait_for_all(self, progress) -> None:
        # No total-duration timeout: uneven workers may legitimately take hours.
        # A dead worker is handled by torchrun; reported application errors abort here.
        last_log = 0.0
        while True:
            failures = list(self.attempt.glob("error-*.json"))
            if failures:
                raise AnnotationError(f"Outro rank falhou: {read_json(failures[0])['error']}")
            if all((self.attempt / f"done-{i}.json").exists() for i in range(self.topology.world_size)):
                return
            if time.monotonic() - last_log > 10:
                progress()
                last_log = time.monotonic()
            time.sleep(0.2)

    def release(self) -> None:
        self.assert_owner()
        shutil.rmtree(self.lock)
        sync_directory(self.output)
