from datetime import datetime, timezone

import pytest

from dante.core.homeostasis import HomeostasisState
from dante.core.relationship import (
    RelationshipModel,
    compute_connection_drive,
    should_initiate_contact,
    update_from_interaction,
)
from dante.core.self_model import SelfModel, build_self_model
from dante.core.values import ValueSystem, get_value_driven_action
from dante.memory.diary import find_contradictions


@pytest.mark.unit
def test_self_model_extracts_patterns_and_increments_version():
    model = build_self_model(
        [
            "Fiquei curioso sobre a memória e uma conexão com Otávio.",
            "A repetição me deixou entediado.",
        ]
    )
    assert model.version == 1
    assert "memória" in model.recurring_themes
    assert model.preferences
    assert model.aversions


@pytest.mark.unit
def test_empty_diary_preserves_existing_self_model():
    previous = SelfModel(version=3)
    assert build_self_model([], previous) is previous


@pytest.mark.unit
def test_relationship_drive_and_interaction_reset():
    model = RelationshipModel(connection_drive=compute_connection_drive(20))
    assert should_initiate_contact(model, HomeostasisState(energy=0.8))
    updated = update_from_interaction(
        model,
        topics=["consciência"],
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert updated.connection_drive == 0
    assert updated.recent_topics == ["consciência"]


@pytest.mark.unit
def test_values_reinforce_and_choose_attention():
    values = ValueSystem()
    values.reinforce("novidade", 0.4)
    assert get_value_driven_action(values) == "explorar uma diferença sem abandonar o contexto"
    assert values.value_history


@pytest.mark.unit
def test_diary_detects_explicit_polarity_change():
    entries = [
        "Eu gosto de observar memória persistente.",
        "Eu não gosto de observar memória persistente.",
    ]
    assert find_contradictions(entries)