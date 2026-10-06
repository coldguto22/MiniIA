"""Decisão explícita de silêncio para evitar geração automática a cada ciclo."""

from __future__ import annotations

from difflib import SequenceMatcher

from dante.core.homeostasis import HomeostasisState
from dante.core.valence import ValenceState


def should_be_silent(
    valence: ValenceState,
    homeostasis: HomeostasisState,
    novelty: float | None = None,
    coherence: float | None = None,
    last_output: str = "",
    current_output: str = "",
    *,
    min_novelty: float = 0.1,
    min_coherence: float = 0.2,
    min_energy: float = 0.2,
    similarity_threshold: float = 0.9,
) -> tuple[bool, str]:
    """Retorna se silenciar e uma razão curta, determinística e auditável."""
    novelty = valence.novelty if novelty is None else max(0.0, min(1.0, novelty))
    coherence = valence.coherence if coherence is None else max(0.0, min(1.0, coherence))
    if homeostasis.energy < min_energy:
        return True, "energia_baixa"
    if novelty < min_novelty:
        return True, "sem_novidade"
    if coherence < min_coherence:
        return True, "sem_coerencia"
    if last_output and current_output and SequenceMatcher(
        None, last_output.strip().casefold(), current_output.strip().casefold()
    ).ratio() >= similarity_threshold:
        return True, "repeticao"
    return False, ""
