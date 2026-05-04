# stability-eval

> Stability-first assertions for LLM prompts and agents. A plugin for [DeepEval](https://github.com/confident-ai/deepeval).

LLM evals tell you if your prompt works **once**. `stability-eval` tells you if it works **every time**.

```python
from stability_eval import stable, cross_model_agreement, perturbation_stable

@stable(runs=5, threshold=1.0)
def test_invoice_extraction():
    ...

@cross_model_agreement(models=["gpt-4o-mini", "claude-opus-4-7", "gemini/gemini-2.0-flash"], threshold=0.85)
def test_classifier_prompt():
    ...

@perturbation_stable(n=10, threshold=0.9)
def test_extraction_robustness():
    ...
```

## Why?

DeepEval and Promptfoo are great at "is this output correct?" Neither makes **stability** a first-class assertion. Most agent failures in production are flakiness, not correctness — and `pass@N` (any of N pass) hides flakiness that `pass^N` (all of N pass) catches.

## Install

```bash
pip install stability-eval
```

The library uses [litellm](https://github.com/BerriAI/litellm) under the hood to talk to LLM providers, so you can use any model you already have access to. Set your API keys as environment variables before running:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
```

## Quick example

```python
import litellm
from stability_eval import stable, cross_model_agreement, perturbation_stable


def extract_total(prompt: str, model: str = "gpt-4o-mini") -> str:
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content


# same prompt, 5 runs, all must pass
@stable(runs=5, threshold=1.0)
def test_extraction_is_deterministic():
    out = extract_total("Extract just the total from: 'Subtotal $1100, tax $134.56. Total: $1,234.56'")
    assert "1,234.56" in out


# GPT, Claude and Gemini must agree on the output
@cross_model_agreement(
    models=["gpt-4o-mini", "claude-haiku-4-5-20251001", "gemini/gemini-2.0-flash"],
    threshold=0.85,
)
def test_extraction_agrees_across_models(model: str):
    return extract_total(
        "Extract just the total from: 'Subtotal $1100, tax $134.56. Total: $1,234.56'",
        model=model,
    )


# reword the prompt 10 ways, output must stay stable
@perturbation_stable(n=10, threshold=0.9)
def test_extraction_robust_to_phrasing(prompt: str):
    return extract_total(prompt)
```

Run with `pytest`.

## Decorators

### `@stable(runs=5, threshold=1.0)`

Runs the test N times. Passes only if `passes / runs >= threshold`. At `threshold=1.0` that's all-or-nothing — useful when you need to be sure a prompt is truly deterministic. Drop it to something like `0.8` if you're okay with one failure in five.

When it fails:

```
AssertionError: @stable failed: 3/5 passed (rate=0.60, required>=1.0)
Failures: ["run 1: AssertionError: assert '1,234.56' in 'The total is $1234.56'",
           "run 3: AssertionError: assert '1,234.56' in 'Total amount: 1234.56 USD'"]
```

### `@cross_model_agreement(models=[...], threshold=0.85, similarity="embedding")`

Calls your function once per model (injects `model=` as a kwarg), then computes pairwise semantic similarity between outputs. Useful for catching prompts that only happen to work well with one model's output style.

`similarity="embedding"` uses `sentence-transformers` locally — fast and no extra API calls. `similarity="judge"` asks an LLM to score the similarity instead, which handles nuance better but is slower and costs money.

When it fails you also get which pair disagreed most and what each model returned, so it's usually obvious what went wrong.

### `@perturbation_stable(n=10, threshold=0.9, judge_model="gpt-4o-mini", prompt_var="prompt")`

Rewrites the prompt N times using `judge_model`, runs your function on each variant, and checks that outputs stay semantically close to the baseline. Good for catching prompts that only work because of a specific phrasing — the kind of thing that breaks when a colleague touches the prompt.

Your function must accept the prompt as a kwarg. The kwarg name defaults to `"prompt"`; change it with `prompt_var` if yours is called something else.

## Works with DeepEval

All three decorators are also exposed as `BaseMetric` subclasses for use inside `assert_test`:

```python
from deepeval import assert_test
from stability_eval.metrics import PassNMetric, CrossModelAgreementMetric, PerturbationStabilityMetric

assert_test(test_case, [
    PassNMetric(runs=5, threshold=1.0),
    CrossModelAgreementMetric(models=["gpt-4o-mini", "claude-haiku-4-5-20251001"], threshold=0.85),
])
```

## License

MIT
