import json
from dataclasses import is_dataclass, asdict


class NISPLogger:
    """
    JSONL logger. Opens the file once and flushes after every write so a mid-run
    crash (OOM, Ctrl-C, timeout) never loses trailing cycles — required for
    publication-quality IEEE runs.

    Use as a context manager:
        with NISPLogger(path) as logger:
            logger.log(payload)
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._fh = None

    def __enter__(self):
        self._fh = open(self.file_path, "a", encoding="utf-8", buffering=1)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def log(self, payload):
        """Accept a dict or a dataclass (CyclePayload). Flush after every write."""
        if is_dataclass(payload):
            payload = asdict(payload)
        line = json.dumps(payload, default=_json_default, allow_nan=False) + "\n"
        if self._fh is None:
            # Fallback for non-context-manager usage — still correct, just slower.
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        else:
            self._fh.write(line)
            self._fh.flush()


def _json_default(o):
    try:
        return list(o)  # numpy arrays, tuples-of-tuples
    except TypeError:
        return str(o)
