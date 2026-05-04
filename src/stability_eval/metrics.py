"""DeepEval BaseMetric subclasses.

Lets users plug stability checks into DeepEval's assert_test() flow alongside
their existing Hallucination, Faithfulness, etc. metrics.
"""

from __future__ import annotations

from typing import Callable, List, Optional

try:
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "stability-eval requires deepeval. Install with: pip install deepeval"
    ) from e

from stability_eval.runners import (
    run_pass_n,
    run_cross_model,
    run_perturbation,
)


class PassNMetric(BaseMetric):
    """pass^N: same prompt, N runs, all (or `threshold` fraction) must pass."""

    def __init__(self, callable_fn: Callable, runs: int = 5, threshold: float = 1.0):
        self.callable_fn = callable_fn
        self.runs = runs
        self.threshold = threshold
        self.score: Optional[float] = None
        self.success: Optional[bool] = None
        self.reason: Optional[str] = None

    def measure(self, test_case: LLMTestCase) -> float:
        result = run_pass_n(self.callable_fn, runs=self.runs, args=(test_case,), kwargs={})
        self.score = result.pass_rate
        self.success = result.pass_rate >= self.threshold
        self.reason = (
            f"{result.passes}/{self.runs} runs passed "
            f"(threshold {self.threshold})"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "PassN Stability"


class CrossModelAgreementMetric(BaseMetric):
    """Cross-model semantic agreement on the same prompt."""

    def __init__(
        self,
        callable_fn: Callable,
        models: List[str],
        threshold: float = 0.85,
        similarity: str = "embedding",
    ):
        self.callable_fn = callable_fn
        self.models = models
        self.threshold = threshold
        self.similarity = similarity
        self.score: Optional[float] = None
        self.success: Optional[bool] = None
        self.reason: Optional[str] = None

    def measure(self, test_case: LLMTestCase) -> float:
        result = run_cross_model(
            self.callable_fn,
            models=self.models,
            similarity=self.similarity,
            args=(test_case,),
            kwargs={},
        )
        self.score = result.min_pairwise_similarity
        self.success = self.score >= self.threshold
        self.reason = (
            f"min pairwise similarity={self.score:.2f} "
            f"({result.disagreement_pair} disagreed most)"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "Cross-Model Agreement"


class PerturbationStabilityMetric(BaseMetric):
    """Stability under semantically equivalent prompt rewordings."""

    def __init__(
        self,
        callable_fn: Callable,
        n: int = 10,
        threshold: float = 0.9,
        judge_model: str = "gpt-4o-mini",
        prompt_var: str = "prompt",
    ):
        self.callable_fn = callable_fn
        self.n = n
        self.threshold = threshold
        self.judge_model = judge_model
        self.prompt_var = prompt_var
        self.score: Optional[float] = None
        self.success: Optional[bool] = None
        self.reason: Optional[str] = None

    def measure(self, test_case: LLMTestCase) -> float:
        result = run_perturbation(
            self.callable_fn,
            n=self.n,
            judge_model=self.judge_model,
            prompt_var=self.prompt_var,
            args=(test_case,),
            kwargs={self.prompt_var: test_case.input},
        )
        self.score = result.mean_similarity
        self.success = self.score >= self.threshold
        self.reason = (
            f"mean similarity across {self.n} rewordings={self.score:.2f}"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "Perturbation Stability"
