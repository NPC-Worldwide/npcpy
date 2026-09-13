"""Run README.md Python examples as pytest tests."""

import os
import re
from pathlib import Path

import pytest


README_PATH = Path(__file__).resolve().parents[1] / "README.md"
SKIP_LONG = os.environ.get("NPC_README_SKIP_LONG", "0").lower() in ("1", "true", "yes")
LONG_INDICES = {5, 23, 24}


def _load_blocks():
    if not README_PATH.exists():
        return []
    readme = README_PATH.read_text()
    blocks = re.findall(r"```python\n(.*?)```", readme, re.DOTALL)
    return [block.strip() for block in blocks if block.strip()]


def _normalize(code):
    code = re.sub(
        r"model\s*=\s*['\"][^'\"]+['\"]",
        "model='kimi-k2.6:cloud'",
        code,
    )
    code = re.sub(
        r"provider\s*=\s*['\"][^'\"]+['\"]",
        "provider='ollama'",
        code,
    )
    return code


def _build_tests():
    blocks = _load_blocks()
    valid_blocks = []
    for idx, block in enumerate(blocks, 1):
        code = _normalize(block)
        try:
            compile(code, f"<readme_example_{idx}>", "exec")
        except SyntaxError:
            continue
        valid_blocks.append((idx, code))

    def test_readme_examples():
        namespace = {}
        for idx, code in valid_blocks:
            if SKIP_LONG and idx in LONG_INDICES:
                print(f"skipping long example {idx}")
                continue
            exec(code, namespace)

    globals()["test_readme_examples"] = test_readme_examples
    return len(valid_blocks)


README_EXAMPLE_COUNT = _build_tests()
