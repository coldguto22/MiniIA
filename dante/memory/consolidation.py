"""Consolidação não destrutiva do diário e esquecimento seletivo auditável."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .diary import find_contradictions, read_entries


@dataclass(frozen=True)
class ConsolidationResult:
    """Resultado de uma consolidação, sem apagar as entradas de origem."""

    chapter: str
    contradictions: list[tuple[str, str]]
    retained_entries: int
    omitted_entries: int


def _entry_is_significant(entry: str) -> bool:
    terms = (
        "descob",
        "curios",
        "conex",
        "coer",
        "consci",
        "memór",
        "relação",
        "otávio",
        "otavio",
        "novid",
    )
    lowered = entry.casefold()
    return len(entry.strip()) >= 40 or any(term in lowered for term in terms)


def select_entries_for_retention(
    entries: Iterable[str], *, limit: int = 50
) -> tuple[list[str], list[str]]:
    """Retém sinais significativos e mantém um limite para evitar crescimento infinito."""
    material = [entry.strip() for entry in entries if entry and entry.strip()]
    significant = [entry for entry in material if _entry_is_significant(entry)]
    retained = (significant or material)[-max(1, limit):]
    retained_ids = {id(entry) for entry in retained}
    omitted = [entry for entry in material if id(entry) not in retained_ids]
    return retained, omitted


def consolidate_entries(
    entries: Iterable[str],
    *,
    summarize: Callable[[str], str] | None = None,
    retention_limit: int = 50,
    now: datetime | None = None,
) -> ConsolidationResult:
    """Resume entradas e identifica contradições sem tratá-las como erro a apagar."""
    material = [entry.strip() for entry in entries if entry and entry.strip()]
    retained, omitted = select_entries_for_retention(material, limit=retention_limit)
    source = "\n---\n".join(retained)
    chapter = summarize(source).strip() if summarize else " ".join(" ".join(item.split()) for item in retained)
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="minutes")
    chapter = f"### Capítulo consolidado ({stamp})\n{chapter[:3000]}"
    return ConsolidationResult(
        chapter=chapter,
        contradictions=find_contradictions(material),
        retained_entries=len(retained),
        omitted_entries=len(omitted),
    )


def consolidate_diary(
    diary_path: str | Path,
    chapter_path: str | Path,
    *,
    summarize: Callable[[str], str] | None = None,
    retention_limit: int = 50,
    now: datetime | None = None,
) -> ConsolidationResult:
    """Consolida o diário e acrescenta um capítulo em arquivo separado."""
    result = consolidate_entries(
        read_entries(diary_path),
        summarize=summarize,
        retention_limit=retention_limit,
        now=now,
    )
    target = Path(chapter_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{result.chapter}\n")
        if result.contradictions:
            handle.write(f"Contradições observadas: {len(result.contradictions)}\n")
    return result