"""Drives homeostáticos simples, persistíveis e independentes de recompensa."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class HomeostasisState:
    """Energia e repetição percebida entre ciclos."""

    energy: float = 1.0
    boredom: float = 0.0
    stagnation_cycles: int = 0
    last_significant: datetime = datetime.min.replace(tzinfo=timezone.utc)
    updated_at: datetime = datetime.min.replace(tzinfo=timezone.utc)

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy", _unit(self.energy))
        object.__setattr__(self, "boredom", _unit(self.boredom))
        object.__setattr__(self, "stagnation_cycles", max(0, int(self.stagnation_cycles)))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("last_significant", "updated_at"):
            data[key] = data[key].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HomeostasisState":
        values = dict(data)
        for key in ("last_significant", "updated_at"):
            if isinstance(values.get(key), str):
                values[key] = datetime.fromisoformat(values[key])
        return cls(**values)


def decay(state: HomeostasisState, *, cycles: int = 1, now: datetime | None = None,
          energy_decay: float = 0.01, boredom_growth: float = 0.02) -> HomeostasisState:
    """Diminui energia e aumenta tédio conforme ciclos passam sem novidade."""
    steps = max(0, int(cycles))
    return HomeostasisState(
        energy=state.energy - max(0.0, energy_decay) * steps,
        boredom=state.boredom + max(0.0, boredom_growth) * steps,
        stagnation_cycles=state.stagnation_cycles + steps,
        last_significant=state.last_significant,
        updated_at=now or datetime.now(timezone.utc),
    )


def replenish(state: HomeostasisState, discovery_type: str,
              amount: float | None = None, *, now: datetime | None = None) -> HomeostasisState:
    """Recupera energia após novidade, descoberta, coerência ou interação."""
    defaults = {"novelty": 0.3, "discovery": 0.5, "coherence": 0.15, "interaction": 0.3}
    recovery = max(0.0, defaults.get(discovery_type, 0.0) if amount is None else amount)
    significant = recovery > 0
    current = now or datetime.now(timezone.utc)
    return HomeostasisState(
        energy=state.energy + recovery,
        boredom=state.boredom - recovery,
        stagnation_cycles=0 if significant else state.stagnation_cycles,
        last_significant=current if significant else state.last_significant,
        updated_at=current,
    )


def rest(state: HomeostasisState, amount: float = 0.01,
         *, now: datetime | None = None) -> HomeostasisState:
    """Recupera energia durante um ciclo passivo sem apagar o tédio acumulado."""
    return HomeostasisState(
        energy=state.energy + max(0.0, amount),
        boredom=state.boredom,
        stagnation_cycles=state.stagnation_cycles,
        last_significant=state.last_significant,
        updated_at=now or datetime.now(timezone.utc),
    )


def should_rest(state: HomeostasisState, threshold: float = 0.2) -> bool:
    return state.energy < threshold


def should_shift_attention(state: HomeostasisState, threshold: float = 0.8) -> bool:
    return state.boredom > threshold


def should_reach_out(state: HomeostasisState, *, now: datetime | None = None,
                     after_hours: float = 24.0, energy_threshold: float = 0.5) -> bool:
    """Indica possível iniciativa de contato; não envia mensagens por conta própria."""
    elapsed = ((now or datetime.now(timezone.utc)) - state.last_significant).total_seconds()
    return elapsed >= max(0.0, after_hours) * 3600 and state.energy > energy_threshold
