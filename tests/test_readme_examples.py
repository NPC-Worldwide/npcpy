"""Run README.md Python examples as pytest tests."""

import os
import re
import traceback
from pathlib import Path


README_PATH = Path(__file__).resolve().parents[1] / "README.md"
SKIP_LONG = os.environ.get("NPC_README_SKIP_LONG", "0").lower() in ("1", "true", "yes")
SKIP_MEDIA = os.environ.get("NPC_README_SKIP_MEDIA", os.environ.get("NPC_README_SKIP_LONG", "0")).lower() in ("1", "true", "yes")
LONG_INDICES = {5, 23, 24}
MEDIA_INDICES = {14}


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
        failures = []
        for idx, code in valid_blocks:
            if SKIP_LONG and idx in LONG_INDICES:
                print(f"skipping long example {idx}")
                continue
            if SKIP_MEDIA and idx in MEDIA_INDICES:
                print(f"skipping media example {idx}")
                continue
            try:
                exec(code, namespace)
                print(f"example {idx}: OK")
            except Exception as e:
                msg = f"example {idx}: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                print(msg)
                failures.append(msg)
        assert not failures, "\n".join(failures)

    globals()["test_readme_examples"] = test_readme_examples
    return len(valid_blocks)


README_EXAMPLE_COUNT = _build_tests()
