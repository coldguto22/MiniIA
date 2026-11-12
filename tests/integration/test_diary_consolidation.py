from datetime import datetime

import pytest

import loop_dante
from dante.memory.consolidation import consolidate_diary


@pytest.mark.integration
def test_consolidate_diary_writes_separate_chapter_without_deleting_source(tmp_path):
    diary_path = tmp_path / "diario.md"
    chapter_path = tmp_path / "diario_capitulos.md"
    entries = "\n".join(
        f"### 01/{index:02d}/2026 10:00\nEntrada significativa sobre memória e novidade {index}."
        for index in range(1, 4)
    )
    diary_path.write_text(entries, encoding="utf-8")

    result = consolidate_diary(
        diary_path,
        chapter_path,
        now=datetime(2026, 10, 6),
    )

    assert result.retained_entries == 3
    assert chapter_path.exists()
    assert diary_path.read_text(encoding="utf-8") == entries
    assert "Capítulo consolidado" in chapter_path.read_text(encoding="utf-8")


@pytest.mark.integration
def test_loop_consolidates_only_at_configured_entry_milestone(tmp_path, monkeypatch):
    diary_path = tmp_path / "diario.md"
    chapter_path = tmp_path / "capitulos.md"
    monkeypatch.setattr(loop_dante, "DIARIO_FILE", str(diary_path))
    monkeypatch.setattr(loop_dante, "CHAPTER_FILE", str(chapter_path))
    monkeypatch.setattr(loop_dante, "CONSOLIDATION_EVERY_ENTRIES", 2)
    diary_path.write_text(
        "### 01/01/2026 10:00\nUma entrada significativa sobre memória.\n"
        "### 02/01/2026 10:00\nOutra entrada significativa sobre novidade.\n",
        encoding="utf-8",
    )

    updated = loop_dante._consolidate_if_needed(2, 0)

    assert updated == 2
    assert chapter_path.exists()