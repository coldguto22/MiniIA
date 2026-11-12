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


@pytest.mark.unit
def test_limitar_texto_modelo_remove_repeticao_e_limita_embedding():
    texto = "Dante: Eu não sei. Dante: Eu não sei. Dante: Eu não sei."

    resultado = loop_dante._limitar_texto_modelo(texto, 40)

    assert resultado == "Dante: Eu não sei."
    assert len(resultado) <= 40


@pytest.mark.unit
def test_loop_uses_modular_reflection_adapter(monkeypatch):
    calls = []

    def fake_reflection(thought, memories, self_model):
        calls.append((thought, list(memories), self_model))
        return "reflexão modular"

    monkeypatch.setattr(loop_dante, "gerar_reflexao_modular", fake_reflection)

    result = loop_dante.gerar_reflexao("pensamento", ["memória relacionada"])

    assert result == "reflexão modular"
    assert calls[0][0] == "pensamento"
    assert calls[0][1] == ["memória relacionada"]
