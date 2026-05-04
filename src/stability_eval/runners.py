"""Execution logic for each stability check.

Decorators are thin wrappers; real work happens here. Keeping them separate
means the metric classes (BaseMetric subclasses) and the decorators can share
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, List, Optional, Tuple

from stability_eval.similarity import embedding_similarity, judge_similarity
from stability_eval.perturb import generate_perturbations


@dataclass
class PassNResult:
    runs: int
    passes: int
    failure_reasons: List[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passes / self.runs if self.runs else 0.0


@dataclass
class CrossModelResult:
    outputs: dict  # model_name -> output
    min_pairwise_similarity: float
    mean_pairwise_similarity: float
    disagreement_pair: Optional[Tuple[str, str]] = None


@dataclass
class PerturbationResult:
    base_output: str
    perturbations: List[Tuple[str, str]]  # (perturbed_prompt, output)
    mean_similarity: float
    worst_perturbation: Optional[str] = None


def run_pass_n(fn: Callable, runs: int, args: tuple, kwargs: dict) -> PassNResult:
    passes = 0
    failures: List[str] = []
    for i in range(runs):
        try:
            fn(*args, **kwargs)
            passes += 1
        except Exception as e:  # noqa: BLE001 — we want to capture all failures
            failures.append(f"run {i}: {type(e).__name__}: {e}")
    return PassNResult(runs=runs, passes=passes, failure_reasons=failures)


def run_cross_model(
    fn: Callable,
    models: List[str],
    similarity: str,
    args: tuple,
    kwargs: dict,
) -> CrossModelResult:
    outputs: dict = {}
    for model in models:
        outputs[model] = fn(*args, model=model, **kwargs)

    sim_fn = embedding_similarity if similarity == "embedding" else judge_similarity

    pairs = list(combinations(models, 2))
    sims = [(a, b, sim_fn(outputs[a], outputs[b])) for a, b in pairs]

    if not sims:
        return CrossModelResult(outputs=outputs, min_pairwise_similarity=1.0,
                                mean_pairwise_similarity=1.0)

    min_sim = min(sims, key=lambda x: x[2])
    mean_sim = sum(s[2] for s in sims) / len(sims)

    return CrossModelResult(
        outputs=outputs,
        min_pairwise_similarity=min_sim[2],
        mean_pairwise_similarity=mean_sim,
        disagreement_pair=(min_sim[0], min_sim[1]),
    )


def run_perturbation(
    fn: Callable,
    n: int,
    judge_model: str,
    prompt_var: str,
    args: tuple,
    kwargs: dict,
) -> PerturbationResult:
    base_prompt = kwargs.get(prompt_var)
    if base_prompt is None:
        raise ValueError(
            f"@perturbation_stable expected kwarg {prompt_var!r} "
            f"holding the prompt text"
        )

    base_output = fn(*args, **kwargs)
    perturbed_prompts = generate_perturbations(base_prompt, n=n, judge_model=judge_model)

    perturbations = []
    for p in perturbed_prompts:
        new_kwargs = {**kwargs, prompt_var: p}
        out = fn(*args, **new_kwargs)
        perturbations.append((p, out))

    # similarity of each perturbed output vs. base
    sims = [embedding_similarity(base_output, out) for _, out in perturbations]
    mean = sum(sims) / len(sims) if sims else 1.0

    worst_idx = min(range(len(sims)), key=lambda i: sims[i]) if sims else None
    worst = perturbations[worst_idx][0] if worst_idx is not None else None

    return PerturbationResult(
        base_output=base_output,
        perturbations=perturbations,
        mean_similarity=mean,
        worst_perturbation=worst,
    )
