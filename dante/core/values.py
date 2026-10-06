"""Valores que se ajustam por experiência, sem objetivo externo."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ValueSystem:
    """Pesos de preferências emergentes e seu histórico observável."""

    values: dict[str, float] = field(
        default_factory=lambda: {"coerência": 0.5, "novidade": 0.5, "relação": 0.5}
    )
    value_history: list[tuple[str, float, str]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        self.values = {key: max(0.0, min(1.0, float(value))) for key, value in self.values.items()}

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["last_updated"] = self.last_updated.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValueSystem":
        values = dict(data)
        if isinstance(values.get("last_updated"), str):
            values["last_updated"] = datetime.fromisoformat(values["last_updated"])
        values["value_history"] = [tuple(item) for item in values.get("value_history", [])]
        return cls(**values)

    def reinforce(self, value_name: str, delta: float, reason: str = "experiência") -> None:
        current = self.values.get(value_name, 0.5)
        updated = max(0.0, min(1.0, current + float(delta)))
        self.values[value_name] = updated
        self.value_history.append((value_name, updated, reason))
        self.value_history = self.value_history[-100:]
        self.last_updated = datetime.now(timezone.utc)


def get_value_driven_action(values: ValueSystem, context: dict[str, Any] | None = None) -> str:
    """Sugere uma atenção possível; não cria obrigação nem tarefa."""
    strongest = max(values.values, key=values.values.get, default="coerência")
    actions = {
        "novidade": "explorar uma diferença sem abandonar o contexto",
        "relação": "prestar atenção ao que pode aprofundar a conexão",
        "coerência": "revisitar memórias antes de concluir",
    }
    return actions.get(strongest, "permanecer atento")