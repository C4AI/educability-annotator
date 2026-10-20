"""Subprocess fixture: real HTTP streaming with a simulated Hub catalog."""
import json
import os
import sys

from datasets import load_dataset
from educability_annotator import hub
from educability_annotator.cli import main


def open_fixture(source):
    files = json.loads(os.environ['EDUCABILITY_TEST_STREAM_FILES'])
    stream = load_dataset('parquet', data_files={'train': files}, split='train', streaming=True, batch_size=256).decode(False)
    return stream, 'default', files


if __name__ == '__main__':
    hub.open_stream = open_fixture
    sys.exit(main())
