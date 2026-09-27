"""Bearer-token API keys for multi-tenant serving.

Keys live in a JSON file mapping key id to {"token" or "sha256": ..., "admin": bool}.
Tokens are held only as SHA-256 digests. Session ids are namespaced by key id
in the server, so a key id must not contain the namespace separator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

NAMESPACE_SEP = "|"
MIN_TOKEN_LEN = 12


@dataclass(frozen=True)
class ApiKey:
    key_id: str
    admin: bool = False


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


class KeyRing:
    def __init__(self, keys: dict[str, tuple[bytes, bool]]) -> None:
        if not keys:
            raise ValueError("key file defines no keys")
        self._by_digest: dict[bytes, ApiKey] = {}
        for key_id, (digest, admin) in keys.items():
            if not key_id or NAMESPACE_SEP in key_id:
                raise ValueError(f"invalid key id {key_id!r}")
            if digest in self._by_digest:
                raise ValueError(f"token for {key_id!r} duplicates another key")
            self._by_digest[digest] = ApiKey(key_id, admin=admin)

    @classmethod
    def from_tokens(cls, tokens: dict[str, tuple[str, bool]]) -> "KeyRing":
        return cls(
            {
                kid: (_checked_digest(kid, tok), adm)
                for kid, (tok, adm) in tokens.items()
            }
        )

    @classmethod
    def from_file(cls, path: str) -> "KeyRing":
        # Each entry holds either the token itself or its hex SHA-256 digest,
        # so an operator can keep the plaintext token out of the file.
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        keys: dict[str, tuple[bytes, bool]] = {}
        for key_id, entry in raw.items():
            if "sha256" in entry:
                digest = bytes.fromhex(entry["sha256"])
            else:
                digest = _checked_digest(key_id, str(entry["token"]))
            keys[key_id] = (digest, bool(entry.get("admin", False)))
        return cls(keys)

    def authenticate(self, authorization: str | None) -> ApiKey | None:
        if not authorization:
            return None
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return None
        return self._by_digest.get(_digest(token.strip()))


def _checked_digest(key_id: str, token: str) -> bytes:
    if len(token) < MIN_TOKEN_LEN:
        raise ValueError(f"token for {key_id!r} is shorter than {MIN_TOKEN_LEN}")
    return _digest(token)
