"""Run Python examples in docs/**/*.md as pytest tests."""

import os
import re
import traceback
from pathlib import Path


DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs"
SKIP_LONG = os.environ.get("NPC_DOCS_SKIP_LONG", os.environ.get("NPC_README_SKIP_LONG", "0")).lower() in ("1", "true", "yes")
SKIP_MEDIA = os.environ.get("NPC_DOCS_SKIP_MEDIA", os.environ.get("NPC_README_SKIP_MEDIA", "0")).lower() in ("1", "true", "yes")
SKIP_ALL = os.environ.get("NPC_DOCS_SKIP_ALL", "0").lower() in ("1", "true", "yes")
LONG_INDICES = {}
MEDIA_INDICES = set()
TEST_MODEL = os.environ.get("NPC_DOCS_TEST_MODEL", "kimi-k2.7-code:cloud")
TEST_PROVIDER = os.environ.get("NPC_DOCS_TEST_PROVIDER", "ollama")


def _load_blocks():
    if not DOCS_ROOT.exists():
        return []
    blocks = []
    for path in sorted(DOCS_ROOT.rglob("*.md")):
        text = path.read_text()
        for block in re.findall(r"```python\n(.*?)```", text, re.DOTALL):
            if block.strip():
                blocks.append((path, block.strip()))
    return blocks


def _normalize(code):
    code = re.sub(
        r"model\s*=\s*['\"][^'\"]+['\"]",
        f"model='{TEST_MODEL}'",
        code,
    )
    code = re.sub(
        r"provider\s*=\s*['\"][^'\"]+['\"]",
        f"provider='{TEST_PROVIDER}'",
        code,
    )
    return code


def _build_tests():
    raw_blocks = _load_blocks()
    valid_blocks = []
    for idx, (path, block) in enumerate(raw_blocks, 1):
        code = _normalize(block)
        try:
            compile(code, f"<doc_example_{idx}>", "exec")
        except SyntaxError:
            continue
        valid_blocks.append((idx, path, code))

    def test_doc_examples():
        if SKIP_ALL:
            print("skipping all doc examples")
            return
        namespace = {}
        failures = []
        for idx, path, code in valid_blocks:
            if SKIP_LONG and idx in LONG_INDICES:
                print(f"skipping long doc example {idx} ({path.name})")
                continue
            if SKIP_MEDIA and idx in MEDIA_INDICES:
                print(f"skipping media doc example {idx} ({path.name})")
                continue
            try:
                exec(code, namespace)
                print(f"doc example {idx} ({path.name}): OK")
            except Exception as e:
                msg = f"doc example {idx} ({path.name}): {type(e).__name__}: {e}\n{traceback.format_exc()}"
                print(msg)
                failures.append(msg)
        assert not failures, "\n".join(failures)

    globals()["test_doc_examples"] = test_doc_examples
    return len(valid_blocks)


DOC_EXAMPLE_COUNT = _build_tests()
