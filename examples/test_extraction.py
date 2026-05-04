"""Example usage. Run with: pytest examples/test_extraction.py"""

import litellm
from stability_eval import stable, cross_model_agreement, perturbation_stable


def extract_invoice_total(prompt: str, model: str = "gpt-4o-mini") -> str:
    """The function under test — a real LLM call."""
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content


# 1) pass^N: same prompt, 5 runs, all must produce "$1,234.56"
@stable(runs=5, threshold=1.0)
def test_extraction_is_deterministic():
    out = extract_invoice_total(
        "Extract just the total from: 'Subtotal $1100, tax $134.56. Total: $1,234.56'"
    )
    assert "1,234.56" in out


# 2) cross-model: GPT, Claude, Gemini must agree
@cross_model_agreement(
    models=["gpt-4o-mini", "claude-haiku-4-5-20251001", "gemini/gemini-2.0-flash"],
    threshold=0.85,
)
def test_extraction_agrees_across_models(model: str):
    return extract_invoice_total(
        "Extract just the total from: 'Subtotal $1100, tax $134.56. Total: $1,234.56'",
        model=model,
    )


# 3) perturbation: reword the prompt 10 ways, output must stay stable
@perturbation_stable(n=10, threshold=0.9)
def test_extraction_robust_to_phrasing(prompt: str):
    return extract_invoice_total(prompt)


if __name__ == "__main__":
    # the perturbation decorator passes prompt as kwarg; show direct call
    test_extraction_robust_to_phrasing(
        prompt="Extract just the total from: 'Subtotal $1100, tax $134.56. Total: $1,234.56'"
    )
