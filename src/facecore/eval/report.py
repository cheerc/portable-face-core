"""Non-biometric report writer with a code redaction guard (Task 10).

The guard runs BEFORE the file is written: any embedding-like float vector,
absolute local path, or per-image identity label fails the whole write.
"""

import re
from pathlib import Path

from facecore.errors import RedactionError

_FLOAT_VECTOR = re.compile(r"\[\s*-?\d+\.\d+(?:\s*,\s*-?\d+\.\d+)+\s*\]")
_ABSOLUTE_PATH = re.compile(r"(?:/Users/|/tmp/|/private/|[A-Za-z]:\\)\S+")
_PER_IMAGE_IDENTITY = re.compile(r"(?i)\bper-image identity\b|\bidentity:\s*enroll-")


def _guard(body: str) -> None:
    if _FLOAT_VECTOR.search(body):
        raise RedactionError("report body carries an embedding-like float vector")
    if _ABSOLUTE_PATH.search(body):
        raise RedactionError("report body carries an absolute local path")
    if _PER_IMAGE_IDENTITY.search(body):
        raise RedactionError("report body carries a per-image identity label")


def write_report(path: Path, header: str, *, extra_body: str = "") -> Path:
    body = header + extra_body
    _guard(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path
