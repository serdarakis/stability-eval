"""Generate semantically equivalent rewordings of a prompt.

Used by @perturbation_stable to fuzz the input phrasing while preserving intent.
A small/cheap model is fine here — we just need varied surface forms.
"""

from __future__ import annotations

import json
from typing import List


def generate_perturbations(prompt: str, n: int = 10, judge_model: str = "gpt-4o-mini") -> List[str]:
    """Return n semantically equivalent rewordings of `prompt`."""
    import litellm

    instruction = (
        f"Reword the following prompt {n} different ways. "
        "Preserve the exact meaning and any required output format. "
        "Vary syntax, word choice, and phrasing — but never change intent. "
        "Reply ONLY with a JSON array of strings, no commentary.\n\n"
        f"Prompt:\n{prompt}"
    )
    resp = litellm.completion(
        model=judge_model,
        messages=[{"role": "user", "content": instruction}],
        temperature=0.7,
    )
    text = resp.choices[0].message.content.strip()

    # tolerate models that wrap in ```json ... ```
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        items = json.loads(text)
        if isinstance(items, list):
            return [str(x) for x in items[:n]]
    except json.JSONDecodeError:
        pass

    # fallback: split lines
    return [line.strip("-• ").strip() for line in text.splitlines() if line.strip()][:n]
