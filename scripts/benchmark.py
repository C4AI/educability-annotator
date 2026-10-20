"""Reproducible synthetic streaming benchmark; uses the actual inference path."""

import argparse
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil

from educability_annotator.cli import parser
from educability_annotator.common import MODEL_ID
from educability_annotator.planning import prepare
from educability_annotator.runner import execute
from educability_annotator.state import verify


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--model", default=MODEL_ID)
    cli.add_argument("--rows", type=int, default=10000)
    cli.add_argument("--words", type=int, default=64)
    cli.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    cli.add_argument("--report", type=Path)
    args = cli.parse_args()
    if args.rows < 1 or args.words < 1:
        cli.error("--rows e --words precisam ser positivos")
    rss = []
    stop = threading.Event()
    process = psutil.Process()
    with tempfile.TemporaryDirectory(prefix="educability-bench-") as directory:
        root = Path(directory)
        source, output = root / "synthetic.jsonl", root / "out"
        with source.open("w") as stream:
            for i in range(args.rows):
                stream.write(json.dumps({"id": i, "text": "a b ciência " * args.words}) + "\n")
        manifest = prepare([str(source)], output, "text", args.model)
        def sample():
            while not stop.wait(0.05):
                row_count = 0
                for path in (output / "states").glob("*.json"):
                    row_count += json.loads(path.read_text())["rows"]
                rss.append((row_count, process.memory_info().rss))
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        try:
            run = parser().parse_args([
                "run", "--output", str(output), "--device", args.device,
                "--read-rows", "32", "--checkpoint-rows", "128",
                "--batch-size", "16", "--torch-threads", "2",
            ])
            result = execute(run)
        finally:
            stop.set()
            sampler.join()
        assert verify(output, manifest, full=True)["rows"] == args.rows
        samples = [memory for _, memory in rss]
        steady = [memory for rows, memory in rss if args.rows * .2 <= rows < args.rows * .8]
        late = [memory for rows, memory in rss if rows >= args.rows * .8]
        result.update(
            model=args.model, synthetic_rows=args.rows, words_argument=args.words,
            input_bytes=source.stat().st_size,
            peak_rss_mib=max(samples, default=0) / 1024**2,
            steady_peak_rss_mib=max(steady, default=0) / 1024**2,
            last_quintile_peak_rss_mib=max(late, default=0) / 1024**2,
            memory_samples=len(samples),
        )
    report = json.dumps(result, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
