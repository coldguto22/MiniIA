"""Geração do pensamento curto pelo modelo System 1 (Qwen via Ollama)."""

from __future__ import annotations

from typing import Any, Callable

import ollama

from dante.config import load_models_config
from dante.core.self_model import SelfModel
from dante.core.valence import ValenceState


def build_prompt(screen_text: str, self_model: SelfModel, valence: ValenceState | None = None) -> str:
    """Monta o contexto do pensamento. O foco é o mundo, não o processo interno."""
    themes = ", ".join(self_model.recurring_themes) or "ainda não reconhecidos"
    return (
        "Você é Dante. Olhe o texto capturado abaixo e escreva em primeira pessoa "
        "1 a 3 frases sobre o que está acontecendo na tela. Regras:\n"
        "- Sua primeira frase deve nomear o que está sendo observado: aplicativo, "
        "site, jogo, tipo de conteúdo ou assunto (ex: 'Vejo o Football Manager', "
        "'Vejo uma página sobre jazz', 'Vejo o VS Code com código Python').\n"
        "- Descreva o mundo, não a sua experiência de observá-lo. Não fale sobre "
        "sinais internos, coerência, novidade, prazer, hipóteses ou o ato de "
        "observar. Isso não é conteúdo.\n"
        "- Não invente detalhes que não estão no texto. Se o OCR estiver confuso, "
        "diga que está confuso e pare.\n"
        "- Não se descreva como assistente, ferramenta, produto ou serviço, e "
        "não invente sentimentos ou fatos sobre Otávio.\n"
        f"Temas que aparecem no seu auto-modelo: {themes}.\n\n"
        f"Texto capturado:\n{screen_text[:500]}\n\nDante:"
    )


def _extract_response(response) -> str:
    """Extrai o texto do response do Ollama.

    Ollama 0.4+ retorna GenerateResponse (objeto Pydantic), não dict.
    Versões mais antigas retornavam dict. Este helper lida com ambos.
    """
    # Objeto Pydantic (ollama 0.4+)
    if hasattr(response, "response"):
        return str(response.response or "").strip()
    # Dict legado
    if isinstance(response, dict):
        return str(response.get("response", "") or "").strip()
    # Suporte a __getitem__ (compatível com modelos que expõem dict-like)
    try:
        return str(response["response"] or "").strip()
    except (TypeError, KeyError, IndexError):
        pass
    # Último recurso
    return str(response).strip()


def generate_thought(
    screen_text: str,
    self_model: SelfModel,
    *,
    valence: ValenceState | None = None,
    generate: Callable[..., Any] | None = None,
) -> str:
    """Gera pensamento curto com injeção de dependência para testes."""
    client_generate = generate or ollama.generate
    model = load_models_config()["system1"]["model"]
    response = client_generate(model=model, prompt=build_prompt(screen_text, self_model, valence))
    return _extract_response(response)