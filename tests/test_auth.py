import hashlib
import json

import pytest

from evoke.auth import (
    ApiKey,
    InvalidKeyId,
    IssuanceDisabled,
    IssuedKeyStore,
    KeyExists,
    KeyRing,
    StaticKey,
)


def _write_keys(tmp_path, payload) -> str:
    path = tmp_path / "keys.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_bearer_token_resolves_to_key(tmp_path):
    ring = KeyRing.from_file(
        _write_keys(
            tmp_path,
            {
                "loom": {"token": "s3cret-loom-key"},
                "ops": {"token": "s3cret-ops-key", "admin": True},
            },
        )
    )
    assert ring.authenticate("Bearer s3cret-loom-key") == ApiKey("loom", admin=False)
    assert ring.authenticate("Bearer s3cret-ops-key") == ApiKey("ops", admin=True)


def test_unknown_missing_or_malformed_credentials_fail(tmp_path):
    ring = KeyRing.from_file(
        _write_keys(tmp_path, {"loom": {"token": "s3cret-loom-key"}})
    )
    assert ring.authenticate(None) is None
    assert ring.authenticate("Bearer nope") is None
    assert ring.authenticate("s3cret-loom-key") is None
    assert ring.authenticate("Basic s3cret-loom-key") is None


def test_rejects_empty_duplicate_or_short_tokens(tmp_path):
    with pytest.raises(ValueError):
        KeyRing.from_file(_write_keys(tmp_path, {}))
    with pytest.raises(ValueError):
        KeyRing.from_file(
            _write_keys(
                tmp_path,
                {"a": {"token": "same-token-1"}, "b": {"token": "same-token-1"}},
            )
        )
    with pytest.raises(ValueError):
        KeyRing.from_file(_write_keys(tmp_path, {"a": {"token": "short"}}))


def test_key_ids_must_not_contain_the_namespace_separator(tmp_path):
    with pytest.raises(ValueError):
        KeyRing.from_file(
            _write_keys(tmp_path, {"lo|om": {"token": "s3cret-loom-key"}})
        )


def test_key_file_may_hold_a_sha256_digest_instead_of_the_token(tmp_path):
    digest = hashlib.sha256(b"s3cret-loom-key").hexdigest()
    ring = KeyRing.from_file(_write_keys(tmp_path, {"loom": {"sha256": digest}}))
    assert ring.authenticate("Bearer s3cret-loom-key") == ApiKey("loom")


def _ring_with_store(tmp_path) -> KeyRing:
    return KeyRing.from_file(
        _write_keys(tmp_path, {"loomd": {"token": "loomd-admin-token", "admin": True}}),
        issued_store=IssuedKeyStore(tmp_path / "issued.json"),
    )


def test_issued_key_authenticates_and_survives_a_restart(tmp_path):
    ring = _ring_with_store(tmp_path)
    token = ring.issue("loom@acme@u1")
    assert ring.authenticate(f"Bearer {token}") == ApiKey("loom@acme@u1")
    assert token not in (tmp_path / "issued.json").read_text()
    reloaded = _ring_with_store(tmp_path)
    assert reloaded.authenticate(f"Bearer {token}") == ApiKey("loom@acme@u1")
    assert [k.key_id for k in reloaded.issued_keys()] == ["loom@acme@u1"]


def test_revoked_key_stops_authenticating_and_stays_revoked(tmp_path):
    ring = _ring_with_store(tmp_path)
    token = ring.issue("loom@acme@u1")
    assert ring.revoke("loom@acme@u1") is True
    assert ring.authenticate(f"Bearer {token}") is None
    assert not ring.is_active("loom@acme@u1")
    assert _ring_with_store(tmp_path).authenticate(f"Bearer {token}") is None
    assert ring.revoke("loom@acme@u1") is False


def test_issue_rejects_bad_duplicate_and_static_ids(tmp_path):
    ring = _ring_with_store(tmp_path)
    ring.issue("loom@acme@u1")
    with pytest.raises(KeyExists):
        ring.issue("loom@acme@u1")
    with pytest.raises(KeyExists):
        ring.issue("loomd")
    with pytest.raises(InvalidKeyId):
        ring.issue("loom|acme")
    with pytest.raises(InvalidKeyId):
        ring.issue("loom/acme")
    with pytest.raises(StaticKey):
        ring.revoke("loomd")


def test_issuance_needs_a_store(tmp_path):
    ring = KeyRing.from_file(
        _write_keys(tmp_path, {"loomd": {"token": "loomd-admin-token"}})
    )
    with pytest.raises(IssuanceDisabled):
        ring.issue("loom@acme@u1")
