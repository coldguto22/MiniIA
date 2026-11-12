import json

import pytest

from dante.models.checkpoint import CheckpointError, ModelCheckpoint, load_checkpoint, save_checkpoint


@pytest.mark.unit
def test_checkpoint_round_trip_and_checksum(tmp_path):
    checkpoint = ModelCheckpoint("mamba", "ollama", "session-a", "ollama-context", [1, 2])
    path = tmp_path / "session.json"
    save_checkpoint(path, checkpoint)

    assert load_checkpoint(path, expected_session_id="session-a") == checkpoint
    data = json.loads(path.read_text(encoding="utf-8"))
    data["payload"].append(3)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CheckpointError, match="checksum"):
        load_checkpoint(path)
