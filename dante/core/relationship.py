"""Modelo evolutivo da relação de Dante com Otávio."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .homeostasis import HomeostasisState


@dataclass
class RelationshipModel:
    """Sinais observados na relação, sem presumir fatos não vividos."""

    otavio_profile: str = "Ainda estou conhecendo Otávio pelas interações que tivemos."
    recent_topics: list[str] = field(default_factory=list)
    last_interaction: datetime = field(default_factory=lambda: datetime.min.replace(tzinfo=timezone.utc))
    connection_drive: float = 0.0
    reciprocity_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.connection_drive = max(0.0, min(1.0, float(self.connection_drive)))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["last_interaction"] = self.last_interaction.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RelationshipModel":
        values = dict(data)
        if isinstance(values.get("last_interaction"), str):
            values["last_interaction"] = datetime.fromisoformat(values["last_interaction"])
        return cls(**values)


def compute_connection_drive(
    time_since_last: float, *, growth_per_hour: float = 0.04
) -> float:
    """Converte horas sem interação em um impulso gradual de conexão."""
    return max(0.0, min(1.0, max(0.0, time_since_last) * max(0.0, growth_per_hour)))


def update_connection_drive(
    model: RelationshipModel,
    *,
    now: datetime | None = None,
    growth_per_hour: float = 0.04,
) -> RelationshipModel:
    """Atualiza o drive com o tempo desde a última interação significativa."""
    current = now or datetime.now(timezone.utc)
    last = model.last_interaction
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    elapsed_hours = max(0.0, (current - last).total_seconds() / 3600)
    model.connection_drive = compute_connection_drive(elapsed_hours, growth_per_hour=growth_per_hour)
    return model


def should_initiate_contact(
    model: RelationshipModel,
    homeostasis: HomeostasisState,
    *,
    threshold: float = 0.7,
    energy_threshold: float = 0.5,
) -> bool:
    """Indica possibilidade de contato; a decisão de expressão permanece externa."""
    return model.connection_drive > threshold and homeostasis.energy > energy_threshold


def update_from_interaction(
    model: RelationshipModel,
    *,
    topics: list[str] | None = None,
    notes: list[str] | None = None,
    profile: str | None = None,
    now: datetime | None = None,
) -> RelationshipModel:
    """Registra uma interação concreta e reduz o drive acumulado.

    Saudações curtas e fragmentos sem conteúdo substantivo NÃO entram no perfil.
    """
    current = now or datetime.now(timezone.utc)

    # Filtro: só aceita tópicos com conteúdo substantivo (>= 5 palavras e
    # pelo menos uma palavra longa, que costuma indicar substantivo ou verbo).
    import re
    def _eh_substantivo(topico: str) -> bool:
        palavras = re.findall(r"\w+", topico, flags=re.UNICODE)
        if len(palavras) < 5:
            return False
        return any(len(p) >= 6 for p in palavras)

    candidatos = list(topics or [])
    substantivos = [t for t in candidatos if _eh_substantivo(t)]

    # Sem tópico substantivo, preserva o perfil anterior intacto.
    if substantivos:
        topic_text = ", ".join(substantivos)
        evolved_profile = profile or model.otavio_profile
        if evolved_profile.startswith("Ainda estou conhecendo"):
            evolved_profile = f"Otávio tem trazido temas como {topic_text}; continuo observando seus interesses."
        novos_topicos = (substantivos or model.recent_topics)[-12:]
    else:
        evolved_profile = profile or model.otavio_profile
        novos_topicos = model.recent_topics

    return RelationshipModel(
        otavio_profile=evolved_profile,
        recent_topics=novos_topicos,
        last_interaction=current,
        connection_drive=0.0,
        reciprocity_notes=(notes or model.reciprocity_notes)[-12:],
    )