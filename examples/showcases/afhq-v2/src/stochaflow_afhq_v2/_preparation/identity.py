"""Canonical serialization used for AFHQ-v2 artifact identities."""

from __future__ import annotations

import hashlib
import json


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
