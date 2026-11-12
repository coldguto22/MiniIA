from pathlib import Path

import pytest

import conversar
import loop_dante
from dante.core.self_model import build_self_model
from dante.core.persistence import load_state
from dante.core.homeostasis import HomeostasisState
from dante.core.valence import ValenceState
from dante.loop import cycle
from dante.core.relationship import RelationshipModel
from dante.core.values import ValueSystem


@pytest.mark.integration
def test_loop_persiste_valencia_e_homeostase_em_ciclos(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    first = cycle("Tela com texto legível e conteúdo relevante para o ciclo.", state_dir=state_dir)
    second = cycle("Tela com texto legível e conteúdo relevante para o ciclo.", state_dir=state_dir)

    assert first.valence.novelty > 0
    assert second.homeostasis.updated_at >= first.homeostasis.updated_at
    assert load_state(state_dir / "valence.json", ValenceState, ValenceState) == second.valence
    assert load_state(state_dir / "homeostasis.json", HomeostasisState, HomeostasisState) == second.homeostasis


@pytest.mark.integration
def test_nova_entrada_do_diario_atualiza_auto_modelo(tmp_path, monkeypatch):
    diary_path = tmp_path / "diario.md"
    model_path = tmp_path / "self_model.json"
    diary_path.write_text(
        "### 06/10/2026 10:00\nFiquei curioso sobre memória e conexão.\n",
        encoding="utf-8",
    )
    from dante.memory.diary import read_entries
    from dante.core.persistence import save_state

    current = build_self_model(read_entries(diary_path))
    save_state(model_path, current)

    assert current.recurring_themes
    assert model_path.exists()


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
