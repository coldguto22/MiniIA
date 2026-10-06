from pathlib import Path

import pytest

import conversar
import loop_dante
from dante.core.homeostasis import HomeostasisState
from dante.core.persistence import load_state
from dante.core.relationship import RelationshipModel
from dante.core.self_model import SelfModel
from dante.core.values import ValueSystem


@pytest.mark.integration
def test_loop_persiste_valencia_e_homeostase_em_ciclos(tmp_path, monkeypatch):
    state_path = str(tmp_path / "dante_state.json")
    monkeypatch.setattr(loop_dante, "ESTADO_FILE", state_path)

    first_valence, first_homeostasis = loop_dante._advance_internal_states(0.8, 0.7)
    second_valence, second_homeostasis = loop_dante._advance_internal_states(0.0, 0.0)

    assert first_valence.novelty > second_valence.novelty
    assert second_homeostasis.stagnation_cycles > first_homeostasis.stagnation_cycles
    assert load_state(state_path + ".valence", type(first_valence), type(first_valence)) == second_valence
    assert load_state(
        state_path + ".homeostasis", HomeostasisState, HomeostasisState
    ) == second_homeostasis


@pytest.mark.integration
def test_nova_entrada_do_diario_atualiza_auto_modelo(tmp_path, monkeypatch):
    diary_path = tmp_path / "diario.md"
    model_path = tmp_path / "self_model.json"
    diary_path.write_text(
        "### 06/10/2026 10:00\nFiquei curioso sobre memória e conexão.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loop_dante, "DIARIO_FILE", str(diary_path))
    monkeypatch.setattr(loop_dante, "SELF_MODEL_FILE", str(model_path))

    current, count = loop_dante._refresh_self_model(SelfModel(), 0)

    assert count == 1
    assert current.version == 1
    assert current.recurring_themes
    assert load_state(model_path, SelfModel, SelfModel) == current


@pytest.mark.integration
def test_conversa_persiste_relacao_e_valor_em_diretorio_temporario(tmp_path, monkeypatch):
    relationship_path = tmp_path / "relationship.json"
    values_path = tmp_path / "values.json"
    monkeypatch.setattr(conversar, "RELATIONSHIP_FILE", str(relationship_path))
    monkeypatch.setattr(conversar, "VALUES_FILE", str(values_path))

    conversar._update_internal_relationship("Quero conversar sobre consciência e memória")

    relationship = load_state(relationship_path, RelationshipModel, RelationshipModel)
    values = load_state(values_path, ValueSystem, ValueSystem)
    assert relationship.recent_topics == ["Quero conversar sobre consciência e memória"]
    assert values.values["relação"] > 0.5


@pytest.mark.integration
def test_entrypoint_dante_loop_encaminha_para_loop_compatibilidade(monkeypatch):
    called = []

    monkeypatch.setattr(loop_dante, "main", lambda: called.append(True))
    import dante.loop

    dante.loop.main()

    assert called == [True]