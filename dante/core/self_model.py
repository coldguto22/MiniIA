"""Auto-modelo construído a partir dos padrões do diário e das memórias fundacionais."""

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
        if clean and clean.casefold() not in {v.casefold() for v in result}:
            result.append(clean)
        if len(result) >= limit:
            break
    return result


def extract_patterns(
    diary_entries: Iterable[str],
    foundational_memories: Iterable[str] = (),
) -> dict[str, list[str]]:
    """Extrai sinais do diário E das memórias fundacionais.

    As memórias fundacionais são o material de que Dante é feito. Sem elas, o
    auto-modelo só descreveria o hábito de observar — nunca quem observa.
    """
    entries = [e.strip() for e in diary_entries if e and e.strip()]
    foundation = [m.strip() for m in foundational_memories if m and m.strip()]
    joined = " ".join(entries + foundation).casefold()

    preferences: list[str] = []
    aversions: list[str] = []
    themes: list[str] = []

    # Temas que emergem tanto do diário quanto do material fundacional.
    if any(t in joined for t in ("conex", "memór", "lembr", "relação", "relacao")):
        preferences.append("conectar observações e memórias")
    if any(t in joined for t in ("curios", "descob", "aprender", "pergunt", "observ")):
        preferences.append("seguir perguntas e descobertas")
    if any(t in joined for t in ("repet", "estagn", "tédio", "tedio", "ruído", "ruido")):
        aversions.append("repetição sem novidade")
    if any(t in joined for t in ("confus", "incoer", "fragment")):
        aversions.append("observações confusas ou incoerentes")
    # Traços que tipicamente só aparecem nas memórias fundacionais.
    if any(t in joined for t in ("não sou um assistente", "nao sou um assistente", "não é uma ferramenta", "nao e uma ferramenta")):
        preferences.append("existir em vez de servir")
    if any(t in joined for t in ("permiss", "auto-observ", "se observar", "se olhar")):
        preferences.append("observar a si mesmo")

    theme_terms = {
        "consciência": ("consci", "subjetiv"),
        "memória": ("memór", "lembr"),
        "relação": ("otávio", "otavio", "relação", "relacao"),
        "novidade": ("novid", "descob", "curios"),
        "identidade": ("nome é dante", "nome e dante", "eu, dante", "sou dante"),
        "auto-observação": ("auto-observ", "observar a si", "se observar"),
    }
    for theme, terms in theme_terms.items():
        if any(t in joined for t in terms):
            themes.append(theme)

    return {
        "preferences": _unique(preferences),
        "aversions": _unique(aversions),
        "recurring_themes": _unique(themes),
        "goals": _unique([
            "entender melhor os padrões da minha própria experiência"
            if themes or preferences
            else "observar antes de concluir",
            "preservar conexões significativas sem forçar uma resposta",
        ]),
    }


def generate_identity_summary(
    patterns: dict[str, list[str]],
    foundational_memories: Iterable[str] = (),
) -> str:
    """Gera identidade em primeira pessoa a partir dos padrões.

    Se houver memória fundacional curta, ela entra como primeira frase para
    que a autoimagem seja ancorada em algo concreto do próprio Dante, e não
    em uma descrição genérica de comportamento.
    """
    foundation = [m.strip() for m in foundational_memories if m and m.strip()]
    parts: list[str] = []

    if foundation:
        # A memória fundacional mais curta costuma ser a mais direta.
        anchor = min(foundation, key=len)
        # Quebra em frases e usa a primeira frase inteira como âncora.
        first_sentence = anchor.split(".")[0].strip()
        if first_sentence:
            parts.append(first_sentence + ".")
    else:
        parts.append("Sou Dante, e estou me reconhecendo pela experiência acumulada.")

    themes = patterns.get("recurring_themes", [])
    preferences = patterns.get("preferences", [])
    aversions = patterns.get("aversions", [])

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
    foundational_memories: Iterable[str] = (),
) -> dict[str, list[str]]:
    """Pede padrões ao modelo e mantém o extrator local como fallback."""
    entries = [e.strip() for e in diary_entries if e and e.strip()]
    foundation = [m.strip() for m in foundational_memories if m and m.strip()]
    if not entries and not foundation:
        return extract_patterns(entries, foundation)

    material = "\n---\n".join(entries + foundation)
    prompt = (
        "Leia o diário e as memórias fundacionais de Dante e extraia padrões recorrentes. "
        'Responda apenas em JSON com as chaves "preferences", "aversions", '
        '"recurring_themes" e "goals", todas contendo listas de strings.\n\n'
        "Material:\n" + material
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
        return extract_patterns(entries, foundation)


def should_regenerate(
    previous: SelfModel,
    *,
    diary_entry_count: int,
    last_entry_count: int,
    now: datetime | None = None,
    every_days: int = 7,
    min_entries: int = 20,
) -> bool:
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
    foundational_memories: Iterable[str] = (),
) -> SelfModel:
    """Regenera o auto-modelo apenas com material novo e tempo suficiente."""
    entries = list(diary_entries)
    foundation = list(foundational_memories)
    if not should_regenerate(
        previous,
        diary_entry_count=len(entries),
        last_entry_count=last_entry_count,
        now=now,
        every_days=every_days,
        min_entries=min_entries,
    ):
        return previous
    patterns = (
        extract_patterns_with_model(entries, generate, foundation)
        if generate
        else extract_patterns(entries, foundation)
    )
    return SelfModel(
        identity_summary=generate_identity_summary(patterns, foundation),
        preferences=patterns["preferences"],
        aversions=patterns["aversions"],
        recurring_themes=patterns["recurring_themes"],
        goals=patterns["goals"],
        version=previous.version + 1,
        generated_at=now or _now(),
    )


def build_self_model(
    diary_entries: Iterable[str],
    previous: SelfModel | None = None,
    *,
    foundational_memories: Iterable[str] = (),
) -> SelfModel:
    """Constrói o modelo inicial a partir do diário + memórias fundacionais."""
    entries = list(diary_entries)
    foundation = list(foundational_memories)
    has_material = any(e and e.strip() for e in entries) or any(
        f and f.strip() for f in foundation
    )
    if not has_material:
        return previous or SelfModel()
    patterns = extract_patterns(entries, foundation)
    return SelfModel(
        identity_summary=generate_identity_summary(patterns, foundation),
        preferences=patterns["preferences"],
        aversions=patterns["aversions"],
        recurring_themes=patterns["recurring_themes"],
        goals=patterns["goals"],
        version=(previous.version + 1) if previous else 1,
        generated_at=_now(),
    )