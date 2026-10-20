"""Bounded preparation/inference/writing pipeline for one task."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .data import PartWriter, read_windows, task_end
from .state import part_path, publish_commit


class Prefetch:
    def __init__(self, iterable, depth: int):
        self.iterable = iterable
        self.queue = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._produce, daemon=True, name="prepare")

    def _put(self, value):
        while not self.stop.is_set():
            try:
                self.queue.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def _produce(self):
        try:
            for item in self.iterable:
                if self.stop.is_set():
                    break
                self._put((item, None, False))
        except BaseException as exc:
            self._put((None, exc, False))
        finally:
            self.iterable.close()
            self._put((None, None, True))

    def __enter__(self):
        self.thread.start()
        return self

    def __iter__(self):
        while True:
            item, error, finished = self.queue.get()
            if error:
                raise error
            if finished:
                return
            yield item

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


class CheckpointWriter:
    def __init__(self, output, task, config, manifest_id, contract_id, state, checkpoint_rows, guard):
        self.output, self.task, self.config = output, task, config
        self.manifest_id, self.contract_id = manifest_id, contract_id
        self.state, self.checkpoint_rows, self.guard = state, checkpoint_rows, guard
        self.writer = None
        self.commit = None

    def _open(self):
        sequence = self.state["next_sequence"]
        self.writer = PartWriter(part_path(self.output, self.task, sequence), self.task, self.config["prefix"])
        self.commit = {
            "manifest_id": self.manifest_id, "contract_id": self.contract_id,
            "task_id": self.task["id"], "sequence": sequence,
            "start": self.state["cursor"], "end": self.state["cursor"], "complete": False,
            **{key: 0 for key in ("rows", "chunks", "tokens", "invalid")},
            **{key: 0.0 for key in ("read_seconds", "tokenize_seconds", "infer_seconds", "write_seconds")},
        }

    def write(self, window, results, read_seconds, tokenize_seconds, infer_seconds):
        start = time.monotonic()
        if self.writer is None:
            self._open()
        self.writer.write(window, results)
        self.commit["rows"] += len(results)
        self.commit["chunks"] += sum(row["chunk_count"] or 0 for row in results)
        self.commit["tokens"] += sum(row["tokens_used"] or 0 for row in results)
        self.commit["invalid"] += sum(row["status"] != "ok" for row in results)
        self.commit["end"] = window.end
        if window.stream_state is not None:
            self.commit["stream_state"] = window.stream_state
        self.commit["complete"] = window.end == task_end(self.task)
        self.commit["read_seconds"] += read_seconds
        self.commit["tokenize_seconds"] += tokenize_seconds
        self.commit["infer_seconds"] += infer_seconds
        self.commit["write_seconds"] += time.monotonic() - start
        if self.commit["rows"] >= self.checkpoint_rows or self.commit["complete"]:
            self._publish()

    def _publish(self):
        self.guard()
        start = time.monotonic()
        self.writer.finish()
        self.writer = None
        self.commit["write_seconds"] += time.monotonic() - start
        self.state = publish_commit(self.output, self.task, self.commit, self.state)
        self.commit = None

    def finish(self):
        if not self.state["complete"]:
            # Empty local inputs and Hub EOF after a full checkpoint need a final part.
            if self.writer is None:
                self._open()
            if task_end(self.task) is None:
                self.commit["complete"] = True
                self.commit.setdefault("stream_state", self.state.get("stream_state"))
            else:
                self.commit["complete"] = self.commit["end"] == task_end(self.task)
            if not self.commit["complete"]:
                raise RuntimeError("Leitura terminou antes de cobrir toda a tarefa.")
            self._publish()

    def abort(self):
        if self.writer is not None:
            self.writer.abort()


def process_task(output, task, manifest, contract_id, state, classifier, args, guard, progress):
    config = manifest["config"]
    writer = CheckpointWriter(
        output, task, config, manifest["manifest_id"], contract_id, state,
        args.checkpoint_rows, guard,
    )

    def prepared_windows():
        windows = iter(read_windows(
            task, state["cursor"], config["text_column"], config["prefix"],
            args.read_rows, args.window_bytes, args.max_record_bytes,
            **({"stream_state": state.get("stream_state")} if task.get("source_kind") == "hub" else {}),
        ))
        try:
            while True:
                start = time.monotonic()
                try:
                    window = next(windows)
                except StopIteration:
                    return
                read_seconds = time.monotonic() - start
                prepared = classifier.prepare(window.texts, config["text_column"])
                yield window, prepared, read_seconds
        finally:
            windows.close()

    pending = deque()
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="write") as executor:
            with Prefetch(prepared_windows(), args.prefetch) as prepared:
                for window, tokens, read_seconds in prepared:
                    # Backpressure bounds both pending prepared data and writer payloads.
                    while len(pending) >= args.prefetch:
                        pending.popleft().result()
                    results, infer_seconds = classifier.predict(tokens)
                    pending.append(executor.submit(
                        writer.write, window, results, read_seconds, tokens.seconds, infer_seconds,
                    ))
                    progress(
                        rows=len(results),
                        chunks=sum(row["chunk_count"] or 0 for row in results),
                        tokens=sum(row["tokens_used"] or 0 for row in results),
                        invalid=sum(row["status"] != "ok" for row in results),
                    )
            while pending:
                pending.popleft().result()
        writer.finish()
        return writer.state
    finally:
        writer.abort()
