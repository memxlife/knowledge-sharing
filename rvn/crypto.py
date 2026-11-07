"""Keys, signatures, hashes and canonical JSON.

A node's id is its Ed25519 public key written as 64 hex characters. That
makes every entry self-checking: anyone can verify a signature from the
author field alone, with no key lookup.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def canonical(obj) -> bytes:
    """The one byte form of a JSON object that every node signs and hashes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_node_id(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


class Identity:
    """The private key a node signs with."""

    def __init__(self, key: Ed25519PrivateKey):
        self._key = key
        raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.node_id = raw.hex()

    @classmethod
    def generate(cls) -> "Identity":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load(cls, path: Path) -> "Identity":
        key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError(f"{path} is not an Ed25519 key")
        return cls(key)

    def save(self, path: Path) -> None:
        pem = self._key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        path = Path(path)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(pem)

    def sign(self, data: bytes) -> str:
        return base64.b64encode(self._key.sign(data)).decode("ascii")


def verify(node_id: str, data: bytes, signature: str) -> bool:
    if not is_node_id(node_id) or not isinstance(signature, str):
        return False
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(node_id))
        key.verify(base64.b64decode(signature, validate=True), data)
        return True
    except (InvalidSignature, ValueError):
        return False


def sign_object(identity: Identity, obj: dict) -> dict:
    """Return a copy of obj with a `signature` over everything else in it."""
    unsigned = {k: v for k, v in obj.items() if k != "signature"}
    return {**unsigned, "signature": identity.sign(canonical(unsigned))}


def verify_object(node_id: str, obj: dict) -> bool:
    unsigned = {k: v for k, v in obj.items() if k != "signature"}
    return verify(node_id, canonical(unsigned), obj.get("signature"))


def object_hash(obj: dict) -> str:
    """Hash of the whole signed object, signature included."""
    return sha256_hex(canonical(obj))
