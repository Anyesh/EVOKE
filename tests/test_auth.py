import hashlib
import json

import pytest

from evoke.auth import ApiKey, KeyRing


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
