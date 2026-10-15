import json

import pytest

from dante.models.checkpoint import CheckpointError, ModelCheckpoint, load_checkpoint, save_checkpoint


@pytest.mark.unit
def test_checkpoint_round_trip_preserves_metadata_and_payload(tmp_path):
    original = ModelCheckpoint(
        model_name="test-mamba",
        backend="test",
        session_id="session-a",
        state_type="json-context",
        payload={"tokens": [1, 2, 3]},
        metadata={"config_hash": "abc"},
    )

    path = tmp_path / "session-a" / "checkpoint.json"
    save_checkpoint(path, original)

    restored = load_checkpoint(
        path,
        expected_model_name="test-mamba",
        expected_backend="test",
        expected_session_id="session-a",
    )
    assert restored == original


@pytest.mark.unit
def test_checkpoint_rejects_tampered_payload(tmp_path):
    path = tmp_path / "checkpoint.json"
    save_checkpoint(
        path,
        ModelCheckpoint("test-mamba", "test", "session-a", "json-context", {"tokens": [1]}),
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["payload"]["tokens"].append(2)
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(CheckpointError, match="checksum"):
        load_checkpoint(path)


@pytest.mark.unit
def test_checkpoint_rejects_other_session(tmp_path):
    path = tmp_path / "checkpoint.json"
    save_checkpoint(
        path,
        ModelCheckpoint("test-mamba", "test", "session-a", "json-context", {"tokens": []}),
    )

    with pytest.raises(CheckpointError, match="session"):
        load_checkpoint(path, expected_session_id="session-b")