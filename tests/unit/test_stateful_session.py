import asyncio

import pytest

from dante.models.checkpoint import CheckpointError, ModelCheckpoint
from dante.models.stateful import ModelSession, SessionStateError


class FakeStatefulModel:
    model_name = "test-mamba"
    backend = "test"

    def __init__(self):
        self.contexts = []

    async def generate_with_state(self, prompt, *, session_id, checkpoint=None, **kwargs):
        context = checkpoint.payload if checkpoint else []
        self.contexts.append((session_id, context))
        return prompt.upper(), ModelCheckpoint(
            model_name=self.model_name,
            backend=self.backend,
            session_id=session_id,
            state_type="test-state",
            payload=[*context, len(prompt)],
        )

    def validate_checkpoint(self, checkpoint, session_id):
        if checkpoint.state_type != "test-state":
            raise CheckpointError("unexpected state type")


@pytest.mark.unit
def test_session_persists_and_resets_state(tmp_path):
    model = FakeStatefulModel()
    session = ModelSession(model, "session-a")

    assert asyncio.run(session.generate("one")) == "ONE"
    checkpoint_path = tmp_path / "session.json"
    session.save(checkpoint_path)
    session.reset()
    assert session.checkpoint is None

    session.restore(checkpoint_path)
    assert asyncio.run(session.generate("two")) == "TWO"
    assert model.contexts == [("session-a", []), ("session-a", [3])]


@pytest.mark.unit
def test_session_rejects_empty_prompt_and_empty_save():
    session = ModelSession(FakeStatefulModel(), "session-a")

    with pytest.raises(ValueError, match="prompt"):
        asyncio.run(session.generate("  "))
    with pytest.raises(SessionStateError, match="without generated state"):
        session.save("unused.json")


@pytest.mark.unit
def test_restore_rejects_another_session(tmp_path):
    model = FakeStatefulModel()
    source = ModelSession(model, "session-a")
    asyncio.run(source.generate("one"))
    path = tmp_path / "session.json"
    source.save(path)

    with pytest.raises(CheckpointError, match="session"):
        ModelSession(model, "session-b").restore(path)
