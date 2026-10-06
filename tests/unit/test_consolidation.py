from datetime import datetime, timezone

import pytest

from dante.memory.consolidation import consolidate_entries, select_entries_for_retention


@pytest.mark.unit
def test_consolidation_retains_significant_entries_and_reports_omissions():
    entries = ["ruído curto"] * 3 + ["Uma descoberta coerente sobre memória persistente e relação."]

    retained, omitted = select_entries_for_retention(entries, limit=10)

    assert retained == [entries[-1]]
    assert len(omitted) == 3


@pytest.mark.unit
def test_consolidation_creates_chapter_and_detects_contradiction():
    entries = [
        "Eu gosto de observar memória persistente e conexão.",
        "Eu não gosto de observar memória persistente e conexão.",
    ]

    result = consolidate_entries(
        entries,
        summarize=lambda source: f"Resumo: {source.splitlines()[0]}",
        now=datetime(2026, 10, 6, tzinfo=timezone.utc),
    )

    assert result.chapter.startswith("### Capítulo consolidado")
    assert result.retained_entries == 2
    assert result.contradictions


@pytest.mark.unit
def test_consolidation_does_not_require_external_model():
    result = consolidate_entries(["Uma entrada significativa sobre novidade e coerência."])

    assert result.chapter
    assert result.omitted_entries == 0