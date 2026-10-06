"""Ciclo testável de observação e compatibilidade com o loop legado."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from dante.cognition.silence import should_be_silent
from dante.core.homeostasis import HomeostasisState, decay, replenish, rest
from dante.core.persistence import load_state, save_state
from dante.core.self_model import SelfModel
from dante.core.valence import compute_coherence, compute_novelty, update_valence
from dante.core.valence import ValenceState
from dante.core.values import ValueSystem


@dataclass
class CycleResult:
    """Resultado observável de um ciclo, incluindo o motivo do silêncio."""

    silent: bool
    reason: str
    thought: str
    reflection: str
    valence: ValenceState
    homeostasis: HomeostasisState
    values: ValueSystem


ThoughtGenerator = Callable[[str, SelfModel, ValenceState], str]
ReflectionGenerator = Callable[[str, Iterable[str], SelfModel], str]


def _state_paths(state_dir: str | Path) -> tuple[Path, Path, Path]:
    directory = Path(state_dir)
    return directory / "valence.json", directory / "homeostasis.json", directory / "values.json"


def cycle(
    screen_text: str | None,
    *,
    related_memories: Iterable[str] = (),
    related_similarities: Iterable[float] = (),
    thought_generator: ThoughtGenerator | None = None,
    reflection_generator: ReflectionGenerator | None = None,
    self_model: SelfModel | None = None,
    last_output: str = "",
    state_dir: str | Path = "dante_state",
    action_initiated: bool = False,
    now: datetime | None = None,
) -> CycleResult:
    """Executa observação, valência, silêncio e expressão sem efeitos globais."""
    current = now or datetime.now(timezone.utc)
    valence_path, homeostasis_path, values_path = _state_paths(state_dir)
    valence = load_state(valence_path, ValenceState, ValenceState)
    homeostasis = load_state(homeostasis_path, HomeostasisState, HomeostasisState)
    values = load_state(values_path, ValueSystem, ValueSystem)
    model = self_model or SelfModel()
    text = (screen_text or "").strip()
    similarities = list(related_similarities)

    if not text:
        valence = update_valence(valence, now=current)
        homeostasis = rest(decay(homeostasis, now=current), now=current)
        silent, reason = should_be_silent(valence, homeostasis, novelty=0.0, coherence=0.0)
        _save_states(valence_path, homeostasis_path, values_path, valence, homeostasis, values)
        return CycleResult(silent, reason, "", "", valence, homeostasis, values)

    novelty = compute_novelty(max(similarities) if similarities else None)
    coherence = compute_coherence(similarities) if similarities else 0.5
    valence = update_valence(
        valence,
        novelty=novelty,
        coherence=coherence,
        agency=1.0 if action_initiated else 0.0,
        now=current,
    )
    homeostasis = decay(homeostasis, now=current)
    if novelty >= 0.5:
        homeostasis = replenish(homeostasis, "novelty", now=current)
    values.reinforce_from_valence(valence)
    silent, reason = should_be_silent(
        valence,
        homeostasis,
        novelty=novelty,
        coherence=coherence,
        last_output=last_output,
    )
    if silent:
        _save_states(valence_path, homeostasis_path, values_path, valence, homeostasis, values)
        return CycleResult(True, reason, "", "", valence, homeostasis, values)

    thought = thought_generator(text, model, valence) if thought_generator else text
    reflection = ""
    if reflection_generator and (valence.pleasure > 0.2 or coherence < 0.4):
        reflection = reflection_generator(thought, related_memories, model)
    _save_states(valence_path, homeostasis_path, values_path, valence, homeostasis, values)
    return CycleResult(False, "", thought, reflection, valence, homeostasis, values)


def _save_states(
    valence_path: Path,
    homeostasis_path: Path,
    values_path: Path,
    valence: ValenceState,
    homeostasis: HomeostasisState,
    values: ValueSystem,
) -> None:
    save_state(valence_path, valence)
    save_state(homeostasis_path, homeostasis)
    save_state(values_path, values)


def main() -> None:
    """Mantém o entry point antigo enquanto a captura é migrada gradualmente."""
    from loop_dante import main as legacy_main

    legacy_main()
