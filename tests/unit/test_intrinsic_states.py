from datetime import datetime, timezone

import pytest

from dante.cognition.silence import should_be_silent
from dante.core.homeostasis import (
    HomeostasisState,
    decay,
    replenish,
    should_reach_out,
    should_rest,
    should_shift_attention,
)
from dante.core.valence import (
    ValenceState,
    compute_agency,
    compute_coherence,
    compute_novelty,
    get_modulation,
    update_valence,
)


@pytest.mark.unit
def test_novelty_coherence_and_agency_are_bounded():
    assert compute_novelty(None) == 1
    assert compute_novelty(0.8) == pytest.approx(0.2)
    assert compute_coherence([0.2, 0.8]) == pytest.approx(0.5)
    assert compute_coherence([]) == 0
    assert compute_agency(True) == 1
    assert compute_agency(False) == 0


@pytest.mark.unit
def test_valence_update_decays_and_modulation_is_bounded():
    earlier = datetime(2026, 1, 1, tzinfo=timezone.utc)
    state = ValenceState(novelty=1, coherence=0.8, timestamp=earlier)
    updated = update_valence(state, novelty=0, coherence=0, decay_rate=0.5)
    assert updated.novelty < state.novelty
    assert updated.coherence < state.coherence
    assert 0 <= get_modulation(updated)["temperature"] <= 1


@pytest.mark.unit
def test_homeostasis_decay_replenish_and_thresholds():
    state = decay(HomeostasisState(energy=0.21, boredom=0.79), energy_decay=0.02)
    assert should_rest(state)
    assert should_shift_attention(state)
    replenished = replenish(state, "discovery")
    assert replenished.energy > state.energy
    assert replenished.stagnation_cycles == 0


@pytest.mark.unit
def test_silence_reports_clear_reasons_and_detects_repetition():
    state = HomeostasisState()
    assert should_be_silent(ValenceState(), state)[1] == "sem_novidade"
    energetic = ValenceState(novelty=0.9, coherence=0.9)
    assert should_be_silent(energetic, HomeostasisState(energy=0.1))[1] == "energia_baixa"
    assert should_be_silent(
        energetic, state, last_output="uma reflexão", current_output="uma reflexão"
    )[1] == "repeticao"


@pytest.mark.unit
def test_relationship_drive_is_only_a_signal():
    old = HomeostasisState(energy=0.8, last_significant=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert should_reach_out(old, now=datetime(2026, 1, 3, tzinfo=timezone.utc))
