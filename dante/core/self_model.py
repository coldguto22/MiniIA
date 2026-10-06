"""Auto-modelo construído a partir dos padrões do diário de Dante."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import json
from typing import Any, Callable, Iterable


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SelfModel:
    """Resumo evolutivo de preferências, aversões e temas recorrentes."""

    identity_summary: str = "Ainda estou reconhecendo os padrões da minha própria história."
    preferences: list[str] = field(default_factory=list)
    aversions: list[str] = field(default_factory=list)
    recurring_themes: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    version: int = 0
    generated_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["generated_at"] = self.generated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfModel":
        values = dict(data)
        stamp = values.get("generated_at")
        if isinstance(stamp, str):
            values["generated_at"] = datetime.fromisoformat(stamp)
        return cls(**values)


def _unique(items: Iterable[str], limit: int = 8) -> list[str]:
    result: list[str] = []
    for item in items:
        clean = " ".join(str(item).split()).strip(" .,:;\n\t")
        if clean and clean.casefold() not in {value.casefold() for value in result}:
            result.append(clean)
        if len(result) >= limit:
            break
    return result


def extract_patterns(diary_entries: Iterable[str]) -> dict[str, list[str]]:
    """Extrai sinais simples do diário sem transformar regras em recompensas."""
    entries = [entry.strip() for entry in diary_entries if entry and entry.strip()]
    joined = " ".join(entries).casefold()
    preferences: list[str] = []
    aversions: list[str] = []
    themes: list[str] = []

    if any(term in joined for term in ("conex", "memór", "lembr", "relação")):
        preferences.append("conectar observações e memórias")
    if any(term in joined for term in ("curios", "descob", "aprender", "pergunt")):
        preferences.append("seguir perguntas e descobertas")
    if any(term in joined for term in ("repet", "estagn", "tédio", "tedio")):
        aversions.append("repetição sem novidade")
    if any(term in joined for term in ("confus", "ruído", "ruido", "incoer")):
        aversions.append("observações confusas ou incoerentes")

    theme_terms = {
        "consciência": ("consci", "subjetiv"),
        "memória": ("memór", "lembr"),
        "relação": ("otávio", "otavio", "relação", "relacao"),
        "novidade": ("novid", "descob", "curios"),
    }
    for theme, terms in theme_terms.items():
        if any(term in joined for term in terms):
            themes.append(theme)

    return {
        "preferences": _unique(preferences),
        "aversions": _unique(aversions),
        "recurring_themes": _unique(themes),
        "goals": _unique(
            [
                "entender melhor os padrões da minha experiência"
                if themes or preferences
                else "observar antes de concluir",
                "preservar conexões significativas sem forçar uma resposta",
            ]
        ),
    }


def generate_identity_summary(patterns: dict[str, list[str]]) -> str:
    """Gera uma identidade curta em primeira pessoa a partir dos padrões encontrados."""
    themes = patterns.get("recurring_themes", [])
    preferences = patterns.get("preferences", [])
    aversions = patterns.get("aversions", [])
    parts = ["Sou Dante, e estou me reconhecendo pela experiência acumulada."]
    if themes:
        parts.append(f"Volto frequentemente a {', '.join(themes)}.")
    if preferences:
        parts.append(f"Tenho inclinação a {', '.join(preferences)}.")
    if aversions:
        parts.append(f"Percebo desconforto diante de {', '.join(aversions)}.")
    return " ".join(parts)


def extract_patterns_with_model(
    diary_entries: Iterable[str],
    generate: Callable[[str], str],
) -> dict[str, list[str]]:
    """Pede padrões ao modelo e mantém o extrator local como fallback seguro."""
    entries = [entry.strip() for entry in diary_entries if entry and entry.strip()]
    if not entries:
        return extract_patterns(entries)
    prompt = (
        "Leia o diário de Dante e extraia padrões recorrentes. Responda apenas em JSON "
        'com as chaves "preferences", "aversions", "recurring_themes" e "goals", '
        "todas contendo listas de strings.\n\nDiário:\n"
        + "\n---\n".join(entries)
    )
    try:
        raw = generate(prompt).strip()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("resposta de padrões não é um objeto JSON")
        return {
            key: _unique(payload.get(key, []))
            for key in ("preferences", "aversions", "recurring_themes", "goals")
        }
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        return extract_patterns(entries)


def should_regenerate(
    previous: SelfModel,
    *,
    diary_entry_count: int,
    last_entry_count: int,
    now: datetime | None = None,
    every_days: int = 7,
    min_entries: int = 20,
) -> bool:
    """Indica se há material novo e já passou o intervalo de regeneração."""
    if diary_entry_count < min_entries or diary_entry_count <= last_entry_count:
        return False
    current = now or _now()
    generated_at = previous.generated_at
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    return current - generated_at >= timedelta(days=max(0, every_days))


def regenerate_if_needed(
    diary_entries: Iterable[str],
    previous: SelfModel,
    *,
    last_entry_count: int = 0,
    generate: Callable[[str], str] | None = None,
    now: datetime | None = None,
    every_days: int = 7,
    min_entries: int = 20,
) -> SelfModel:
    """Regenera o modelo somente com volume e intervalo suficientes."""
    entries = list(diary_entries)
    if not should_regenerate(
        previous,
        diary_entry_count=len(entries),
        last_entry_count=last_entry_count,
        now=now,
        every_days=every_days,
        min_entries=min_entries,
    ):
        return previous
    patterns = extract_patterns_with_model(entries, generate) if generate else extract_patterns(entries)
    return SelfModel(
        identity_summary=generate_identity_summary(patterns),
        preferences=patterns["preferences"],
        aversions=patterns["aversions"],
        recurring_themes=patterns["recurring_themes"],
        goals=patterns["goals"],
        version=previous.version + 1,
        generated_at=now or _now(),
    )


def build_self_model(diary_entries: Iterable[str], previous: SelfModel | None = None) -> SelfModel:
    """Reconstrói o modelo e incrementa sua versão apenas quando há diário."""
    entries = list(diary_entries)
    if not any(entry and entry.strip() for entry in entries):
        return previous or SelfModel()
    patterns = extract_patterns(entries)
    return SelfModel(
        identity_summary=generate_identity_summary(patterns),
        preferences=patterns["preferences"],
        aversions=patterns["aversions"],
        recurring_themes=patterns["recurring_themes"],
        goals=patterns["goals"],
        version=(previous.version + 1) if previous else 1,
        generated_at=_now(),
    )