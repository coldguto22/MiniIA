"""Leitura e consolidação do diário persistente de Dante."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path


_ENTRY_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)


def read_entries(path: str | Path) -> list[str]:
    """Retorna as entradas do diário sem modificar o arquivo original."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    markers = list(_ENTRY_RE.finditer(text))
    if not markers:
        return [text.strip()] if text.strip() else []
    entries: list[str] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        body = text[marker.end():end].strip()
        if body:
            entries.append(body)
    return entries


def consolidate_old_entries(
    path: str | Path = "diario.md", *, days_threshold: int = 30, now: datetime | None = None
) -> str:
    """Resume títulos e conteúdo antigo sem apagar evidência do diário."""
    cutoff = (now or datetime.now()) - timedelta(days=max(0, days_threshold))
    old: list[str] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""
    for marker in _ENTRY_RE.finditer(text):
        date_text = marker.group(1).split("(")[0].strip()
        try:
            entry_date = datetime.strptime(date_text, "%d/%m/%Y %H:%M")
        except ValueError:
            continue
        if entry_date >= cutoff:
            continue
        end = len(text)
        following = _ENTRY_RE.search(text, marker.end())
        if following:
            end = following.start()
        old.append(text[marker.end():end].strip())
    if not old:
        return ""
    summary = " ".join(" ".join(item.split()) for item in old)
    return f"Capítulo anterior ({len(old)} entradas): {summary[:1000]}"


def find_contradictions(entries: list[str]) -> list[tuple[str, str]]:
    """Encontra pares explícitos de afirmação e negação sobre o mesmo termo."""
    contradictions: list[tuple[str, str]] = []
    for index, first in enumerate(entries):
        first_words = set(re.findall(r"[a-záàâãéêíóôõúç]{5,}", first.casefold()))
        for second in entries[index + 1:]:
            second_words = set(re.findall(r"[a-záàâãéêíóôõúç]{5,}", second.casefold()))
            shared = first_words & second_words
            if not shared:
                continue
            first_negative = any(term in first.casefold() for term in ("não ", "nunca ", "jamais "))
            second_negative = any(term in second.casefold() for term in ("não ", "nunca ", "jamais "))
            if first_negative != second_negative:
                contradictions.append((first, second))
    return contradictions