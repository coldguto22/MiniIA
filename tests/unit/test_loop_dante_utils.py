import pytest

import loop_dante


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
