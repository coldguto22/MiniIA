"""Valência intrínseca calculada a partir de sinais cognitivos observáveis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import exp
from typing import Any


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class ValenceState:
    """Snapshot de valência; prazer aceita valores de -1 a 1."""

    pleasure: float = 0.0
    arousal: float = 0.0
    coherence: float = 0.0
    novelty: float = 0.0
    agency: float = 0.0
    timestamp: datetime = datetime.min.replace(tzinfo=timezone.utc)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pleasure", max(-1.0, min(1.0, float(self.pleasure))))
        for name in ("arousal", "coherence", "novelty", "agency"):
            object.__setattr__(self, name, _unit(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValenceState":
        values = dict(data)
        stamp = values.get("timestamp")
        if isinstance(stamp, str):
            values["timestamp"] = datetime.fromisoformat(stamp)
        return cls(**values)


def compute_novelty(max_similarity: float | None) -> float:
    """Converte similaridade cosseno (0..1) da memória em novidade."""
    return 1.0 if max_similarity is None else _unit(1.0 - max_similarity)


def compute_coherence(similarities: list[float]) -> float:
    """Estima coerência pela similaridade média com memórias relacionadas."""
    return _unit(sum(similarities) / len(similarities)) if similarities else 0.0


def compute_agency(action_initiated: bool) -> float:
    """Representa se a operação foi iniciada internamente ou por input externo."""
    return 1.0 if action_initiated else 0.0


def update_valence(
    previous: ValenceState | None = None,
    *,
    novelty: float = 0.0,
    coherence: float = 0.0,
    agency: float = 0.0,
    decay_rate: float = 0.05,
    now: datetime | None = None,
) -> ValenceState:
    """Atualiza sinais com decaimento exponencial, sem recompensa de tarefa."""
    now = now or datetime.now(timezone.utc)
    prior = previous or ValenceState()
    decay = exp(-max(0.0, decay_rate))
    n, c, a = _unit(novelty), _unit(coherence), _unit(agency)
    pleasure = max(-1.0, min(1.0, prior.pleasure * decay + (n + c + a - 1.5) * (1 - decay)))
    return ValenceState(
        pleasure=pleasure,
        arousal=_unit(prior.arousal * decay + n * (1 - decay)),
        coherence=_unit(prior.coherence * decay + c * (1 - decay)),
        novelty=_unit(prior.novelty * decay + n * (1 - decay)),
        agency=_unit(prior.agency * decay + a * (1 - decay)),
        timestamp=now,
    )


def get_modulation(state: ValenceState) -> dict[str, float]:
    """Retorna ajustes graduais para os próximos ciclos cognitivos."""
    return {
        "temperature": max(0.2, min(1.0, 0.5 + state.arousal * 0.3 + state.novelty * 0.2)),
        "curiosity_threshold": max(0.1, min(0.9, 0.7 - state.novelty * 0.25 - state.pleasure * 0.1)),
        "verbosity": max(0.0, min(1.0, 0.3 + state.coherence * 0.4 + state.arousal * 0.3)),
    }
