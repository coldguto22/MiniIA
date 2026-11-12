from datetime import datetime, timezone

import pytest

import loop_dante
from dante.memory.consolidation import consolidate_diary, consolidate_entries


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
def test_consolidation_keeps_source_and_respects_retention_limit(tmp_path):
    diary_path = tmp_path / "diario.md"
    chapter_path = tmp_path / "capitulos.md"
    entries = ["Entrada antiga sem relevância " + ("x" * 50), "Memória e novidade são temas recorrentes."]
    original = "\n".join(entries)
    diary_path.write_text(original, encoding="utf-8")
    result = consolidate_entries(entries, retention_limit=1, now=datetime(2026, 10, 6, tzinfo=timezone.utc))
    saved = consolidate_diary(diary_path, chapter_path, retention_limit=1)

    assert result.retained_entries == 1
    assert result.omitted_entries == 1
    assert chapter_path.exists()
    assert saved.retained_entries == 1
    assert diary_path.read_text(encoding="utf-8") == original
