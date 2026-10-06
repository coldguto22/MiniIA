from datetime import datetime, timezone

import pytest

from dante.core.persistence import load_state
from dante.core.valence import ValenceState
from dante.loop import cycle


@pytest.mark.unit
def test_cycle_silences_empty_observation_and_persists_state(tmp_path):
    result = cycle(
        "",
        state_dir=tmp_path,
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert result.silent is True
    assert result.reason == "sem_novidade"
    assert load_state(tmp_path / "valence.json", ValenceState, ValenceState) == result.valence


@pytest.mark.unit
def test_cycle_calls_generators_only_when_observation_is_significant(tmp_path):
    calls = []

    def thought(text, model, valence):
        calls.append(("thought", text))
        return "pensamento próprio"

    def reflection(text, memories, model):
        calls.append(("reflection", text))
        return "reflexão própria"

    result = cycle(
        "Uma observação nova e legível apareceu na tela.",
        related_memories=["memória anterior"],
        related_similarities=[0.1, 0.5],
        thought_generator=thought,
        reflection_generator=reflection,
        action_initiated=True,
        state_dir=tmp_path,
    )

    assert result.silent is False
    assert result.thought == "pensamento próprio"
    assert result.reflection == "reflexão própria"
    assert calls == [("thought", "Uma observação nova e legível apareceu na tela."), ("reflection", "pensamento próprio")]