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
