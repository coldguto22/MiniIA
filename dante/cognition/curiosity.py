"""Curiosidade modulada por novidade, valência e valores emergentes."""

from __future__ import annotations

from dante.core.valence import ValenceState, get_modulation
from dante.core.values import ValueSystem


def compute_curiosity(
    text: str,
    valence: ValenceState,
    values: ValueSystem | None = None,
) -> float:
    """Calcula um sinal de curiosidade, sem transformar pesquisa em obrigação."""
    if len(text.strip()) < 30:
        return 0.0
    novelty = valence.novelty
    relational_weight = (values.values.get("novidade", 0.5) if values else 0.5)
    arousal = valence.arousal
    return max(0.0, min(1.0, novelty * 0.55 + arousal * 0.25 + relational_weight * 0.2))


def curiosity_threshold(valence: ValenceState) -> float:
    """Expõe o limiar modulado para tornar a decisão auditável no loop."""
    return get_modulation(valence)["curiosity_threshold"]