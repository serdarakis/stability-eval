"""Decorator API for stability assertions.

These are the user-facing entry points. They wrap a test function and either
pass through to pytest (causing test failure on instability) or can be invoked
directly to get a structured result.
"""

from __future__ import annotations

import functools
from typing import Callable, List, Optional

from stability_eval.runners import (
    run_pass_n,
    run_cross_model,
    run_perturbation,
)


def stable(runs: int = 5, threshold: float = 1.0):
    """Run the wrapped test `runs` times. Pass only if pass-rate >= threshold.

    threshold=1.0 implements the strict pass^N metric (all runs must pass).
    threshold=0.8 with runs=5 means at least 4/5 must pass.

    The wrapped function should raise (e.g. assertion error) on a failed run.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = run_pass_n(fn, runs=runs, args=args, kwargs=kwargs)
            if result.pass_rate < threshold:
                raise AssertionError(
                    f"@stable failed: {result.passes}/{runs} passed "
                    f"(rate={result.pass_rate:.2f}, required>={threshold})\n"
                    f"Failures: {result.failure_reasons}"
                )
            return result
        return wrapper
    return decorator


def cross_model_agreement(
    models: List[str],
    prompt_var: str = "prompt",
    threshold: float = 0.85,
    similarity: str = "embedding",
):
    """Call each model with the prompt; assert pairwise similarity >= threshold.

    The wrapped function must accept a `model` kwarg and return the LLM output
    as a string. `prompt_var` names the kwarg holding the prompt text (used in
    diagnostics).

    similarity: "embedding" (sentence-transformers cosine) or "judge" (LLM-as-judge).
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = run_cross_model(
                fn, models=models, similarity=similarity,
                args=args, kwargs=kwargs,
            )
            if result.min_pairwise_similarity < threshold:
                raise AssertionError(
                    f"@cross_model_agreement failed: "
                    f"min pairwise similarity={result.min_pairwise_similarity:.2f} "
                    f"(required>={threshold})\n"
                    f"Disagreement: {result.disagreement_pair}\n"
                    f"Outputs: {result.outputs}"
                )
            return result
        return wrapper
    return decorator


def perturbation_stable(
    n: int = 10,
    threshold: float = 0.9,
    judge_model: str = "gpt-4o-mini",
    prompt_var: str = "prompt",
):
    """Reword the prompt `n` ways using `judge_model`, assert outputs are stable.

    The wrapped function must accept the prompt as a kwarg named `prompt_var`
    and return a string output.

    threshold is the minimum mean pairwise semantic similarity between outputs
    across all perturbations.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = run_perturbation(
                fn, n=n, judge_model=judge_model, prompt_var=prompt_var,
                args=args, kwargs=kwargs,
            )
            if result.mean_similarity < threshold:
                raise AssertionError(
                    f"@perturbation_stable failed: "
                    f"mean similarity={result.mean_similarity:.2f} "
                    f"(required>={threshold})\n"
                    f"Most divergent perturbation: {result.worst_perturbation!r}"
                )
            return result
        return wrapper
    return decorator
