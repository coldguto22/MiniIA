import pytest

import loop_dante
from dante.core.homeostasis import HomeostasisState


@pytest.mark.unit
def test_texto_e_ruido_retorna_true_para_texto_curto():
    assert loop_dante.texto_e_ruido("abc") is True


@pytest.mark.unit
def test_texto_e_ruido_retorna_false_para_texto_valido():
    texto = "Dante observa uma interface com texto legivel e contexto coerente para analise"
    assert loop_dante.texto_e_ruido(texto) is False


@pytest.mark.unit
def test_parece_recusa_detecta_padrao():
    assert loop_dante.parece_recusa("Desculpe, mas não posso ajudar com isso") is True


@pytest.mark.unit
def test_parece_recusa_aceita_texto_normal():
    assert loop_dante.parece_recusa("Hoje observei em silencio e refleti sobre o que vi") is False


@pytest.mark.unit
def test_deriva_de_papel_vira_silencio():
    assert loop_dante.parece_deriva_de_papel("Sou um assistente pronto para ajudar")
    assert not loop_dante.parece_deriva_de_papel("Hoje observei uma mudança na tela")


@pytest.mark.unit
def test_diary_cadence_depends_on_energy_or_spaced_milestone():
    assert loop_dante.should_write_diary(HomeostasisState(energy=0.2), 1)
    assert loop_dante.should_write_diary(HomeostasisState(energy=0.8), 20)
    assert not loop_dante.should_write_diary(HomeostasisState(energy=0.8), 2)


@pytest.mark.unit
def test_novidade_reabastece_homeostase_proporcionalmente(tmp_path, monkeypatch):
    state_path = str(tmp_path / "dante_state.json")
    monkeypatch.setattr(loop_dante, "ESTADO_FILE", state_path)
    monkeypatch.setattr(loop_dante, "VALUES_FILE", str(tmp_path / "values.json"))
    monkeypatch.setattr(loop_dante, "RELATIONSHIP_FILE", str(tmp_path / "relationship.json"))

    _, homeostasis = loop_dante._advance_internal_states(0.6, 0.7)

    assert homeostasis.energy == pytest.approx(1.0)
    assert homeostasis.boredom == pytest.approx(0.0)


@pytest.mark.unit
def test_salvar_na_memoria_separa_substancia_e_contexto(monkeypatch):
    class FakeCollection:
        def __init__(self):
            self.added = None

        def query(self, **kwargs):
            return {"distances": [[]], "ids": [[]], "metadatas": [[]]}

        def add(self, **kwargs):
            self.added = kwargs

    collection = FakeCollection()
    monkeypatch.setattr(loop_dante, "colecao", collection)
    monkeypatch.setattr(loop_dante, "gerar_embedding", lambda text: [1.0, 0.0, 0.0])

    loop_dante.salvar_na_memoria(
        "pensamento substancial",
        "observacao_passiva",
        metadata_extra={"observacao_bruta": "texto da tela", "tipo": "nao-permitido"},
    )

    assert collection.added["documents"] == ["pensamento substancial"]
    assert collection.added["metadatas"][0]["observacao_bruta"] == "texto da tela"
    assert collection.added["metadatas"][0]["tipo"] == "observacao_passiva"
