"""Bearer-token API keys for multi-tenant serving.

Static keys come from an operator-owned JSON file mapping key id to
{"token" or "sha256": ..., "admin": bool}. Non-admin keys can also be issued
and revoked at runtime by an admin; those live in a separate server-owned
store that holds only digests, so a restart keeps them valid. Tokens are only
ever compared as SHA-256 digests. Session ids are namespaced by key id in the
server, so key ids exclude the namespace separator and "/" (which delimits
the session header's tenant prefix).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

NAMESPACE_SEP = "|"
MIN_TOKEN_LEN = 12
KEY_ID_RE = re.compile(r"[A-Za-z0-9._@:+-]{1,128}")
ISSUED_TOKEN_PREFIX = "evk_"


class InvalidKeyId(ValueError):
    pass


class KeyExists(Exception):
    pass


class StaticKey(Exception):
    pass


class IssuanceDisabled(Exception):
    pass


@dataclass(frozen=True)
class ApiKey:
    key_id: str
    admin: bool = False


@dataclass(frozen=True)
class IssuedKey:
    key_id: str
    digest: bytes
    created: str


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def _checked_digest(key_id: str, token: str) -> bytes:
    if len(token) < MIN_TOKEN_LEN:
        raise ValueError(f"token for {key_id!r} is shorter than {MIN_TOKEN_LEN}")
    return _digest(token)


def _check_key_id(key_id: str) -> None:
    if not KEY_ID_RE.fullmatch(key_id):
        raise InvalidKeyId(f"key id {key_id!r} must match {KEY_ID_RE.pattern}")


class IssuedKeyStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> dict[str, IssuedKey]:
        if not self._path.exists():
            return {}
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        return {
            key_id: IssuedKey(key_id, bytes.fromhex(entry["sha256"]), entry["created"])
            for key_id, entry in raw.get("keys", {}).items()
        }

    def save(self, keys: dict[str, IssuedKey]) -> None:
        payload = {
            "keys": {
                k.key_id: {"sha256": k.digest.hex(), "created": k.created}
                for k in keys.values()
            }
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash mid-write never leaves a truncated
        # store, which would silently revoke every issued key on restart.
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".issued-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self._path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


class KeyRing:
    def __init__(
        self,
        keys: dict[str, tuple[bytes, bool]],
        *,
        issued_store: IssuedKeyStore | None = None,
    ) -> None:
        if not keys:
            raise ValueError("key file defines no keys")
        self._by_digest: dict[bytes, ApiKey] = {}
        self._static: set[str] = set()
        for key_id, (digest, admin) in keys.items():
            _check_key_id(key_id)
            self._add(ApiKey(key_id, admin=admin), digest)
            self._static.add(key_id)
        self._store = issued_store
        self._issued: dict[str, IssuedKey] = {}
        if issued_store is not None:
            for key in issued_store.load().values():
                if key.key_id in self._static:
                    raise ValueError(f"issued key {key.key_id!r} shadows a static key")
                self._add(ApiKey(key.key_id), key.digest)
                self._issued[key.key_id] = key

    def _add(self, key: ApiKey, digest: bytes) -> None:
        if digest in self._by_digest:
            raise ValueError(f"token for {key.key_id!r} duplicates another key")
        self._by_digest[digest] = key

    @classmethod
    def from_tokens(
        cls,
        tokens: dict[str, tuple[str, bool]],
        *,
        issued_store: IssuedKeyStore | None = None,
    ) -> "KeyRing":
        return cls(
            {
                kid: (_checked_digest(kid, tok), adm)
                for kid, (tok, adm) in tokens.items()
            },
            issued_store=issued_store,
        )

    @classmethod
    def from_file(
        cls, path: str, *, issued_store: IssuedKeyStore | None = None
    ) -> "KeyRing":
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
        return cls(keys, issued_store=issued_store)

    def authenticate(self, authorization: str | None) -> ApiKey | None:
        if not authorization:
            return None
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return None
        return self._by_digest.get(_digest(token.strip()))

    def is_active(self, key_id: str) -> bool:
        return key_id in self._static or key_id in self._issued

    @property
    def issuance_enabled(self) -> bool:
        return self._store is not None

    def issued_keys(self) -> list[IssuedKey]:
        return sorted(self._issued.values(), key=lambda k: k.key_id)

    def static_key_ids(self) -> list[str]:
        return sorted(self._static)

    def issue(self, key_id: str) -> str:
        if self._store is None:
            raise IssuanceDisabled("no issued-key store is configured")
        _check_key_id(key_id)
        if self.is_active(key_id):
            raise KeyExists(key_id)
        token = ISSUED_TOKEN_PREFIX + secrets.token_urlsafe(32)
        key = IssuedKey(
            key_id,
            _digest(token),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._store.save({**self._issued, key_id: key})
        self._issued[key_id] = key
        self._add(ApiKey(key_id), key.digest)
        return token

    def revoke(self, key_id: str) -> bool:
        if key_id in self._static:
            raise StaticKey(key_id)
        key = self._issued.get(key_id)
        if key is None:
            return False
        remaining = {k: v for k, v in self._issued.items() if k != key_id}
        assert self._store is not None
        self._store.save(remaining)
        self._issued = remaining
        del self._by_digest[key.digest]
        return True
