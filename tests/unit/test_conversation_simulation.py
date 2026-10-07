import pytest

import conversar
from dante.core.homeostasis import HomeostasisState
from dante.core.persistence import save_state
from dante.core.valence import ValenceState


@pytest.mark.unit
def test_two_turn_conversation_preserves_history_and_memory_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(conversar, "RELATIONSHIP_FILE", str(tmp_path / "relationship.json"))
    monkeypatch.setattr(conversar, "VALUES_FILE", str(tmp_path / "values.json"))
    monkeypatch.setattr(conversar, "VALENCE_FILE", str(tmp_path / "valence.json"))
    monkeypatch.setattr(conversar, "HOMEOSTASIS_FILE", str(tmp_path / "homeostasis.json"))
    save_state(tmp_path / "valence.json", ValenceState(novelty=0.4, coherence=0.7))
    save_state(tmp_path / "homeostasis.json", HomeostasisState(energy=0.8, boredom=0.2))
    conversar.historico.clear()
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"response": "Não tenho evidência suficiente para afirmar isso."}
        return {"response": "Discordo: essa conclusão não decorre das memórias recuperadas."}

    hits = [
        conversar.MemoryHit(
            "Consciência foi discutida como hipótese.",
            0.18,
            "diario",
            "diario.md",
        )
    ]
    try:
        first, first_prompt = conversar.respond_to(
            "O que você sabe sobre consciência?", hits, generate=fake_generate
        )
        second, second_prompt = conversar.respond_to(
            "Você concorda com a minha conclusão?", hits, generate=fake_generate
        )
        assert len(conversar.historico) == 4
    finally:
        conversar.historico.clear()

    assert first.startswith("Não tenho evidência")
    assert second.startswith("Discordo")
    assert "tipo=diario" in first_prompt
    assert "fonte=diario.md" in first_prompt
    assert first in second_prompt
    assert all(call["model"] == "llama3.1:8b" for call in calls)
    assert all(call["options"]["temperature"] == 0.3 for call in calls)