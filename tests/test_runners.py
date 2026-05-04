"""Unit tests that don't hit any LLM — verify the runner logic itself."""

import pytest
from stability_eval.runners import run_pass_n


def test_pass_n_all_pass():
    def always_passes():
        assert True

    result = run_pass_n(always_passes, runs=5, args=(), kwargs={})
    assert result.passes == 5
    assert result.pass_rate == 1.0


def test_pass_n_some_fail():
    counter = {"i": 0}

    def flaky():
        counter["i"] += 1
        if counter["i"] % 2 == 0:
            raise AssertionError("flake")

    result = run_pass_n(flaky, runs=4, args=(), kwargs={})
    assert result.passes == 2
    assert result.pass_rate == 0.5
    assert len(result.failure_reasons) == 2


def test_pass_n_decorator_raises_on_threshold():
    from stability_eval import stable

    @stable(runs=3, threshold=1.0)
    def always_fails():
        raise AssertionError("nope")

    with pytest.raises(AssertionError, match="@stable failed"):
        always_fails()
