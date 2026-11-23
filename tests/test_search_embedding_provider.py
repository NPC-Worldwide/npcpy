"""search_similar_texts must embed with the requested embedding_provider."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy  # noqa: F401  (import before sys.modules is patched)
import pytest


@pytest.fixture
def search_module():
    calls = []

    def fake_get_embeddings(texts, model, provider):
        calls.append((list(texts), model, provider))
        return [[1.0, 0.0] if "match" in t else [0.0, 1.0] for t in texts]

    path = Path(__file__).parents[1] / "npcpy" / "memory" / "search.py"
    spec = importlib.util.spec_from_file_location("memory_search_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "npcpy.data.load": SimpleNamespace(load_file_contents=Mock()),
        "npcpy.data.web": SimpleNamespace(search_web=Mock()),
        "npcpy.gen.embeddings": SimpleNamespace(
            get_embeddings=fake_get_embeddings,
            get_ollama_embeddings=Mock(side_effect=AssertionError("ollama used")),
        ),
        "npcpy.llm_funcs": SimpleNamespace(get_llm_response=Mock()),
        "npcpy.npc_sysenv": SimpleNamespace(render_markdown=Mock()),
    }):
        spec.loader.exec_module(module)
    return module, calls


def test_docs_path_uses_embedding_provider(search_module):
    module, calls = search_module
    out = module.search_similar_texts(
        "match query", "text-embedding-3-small", "openai",
        docs_to_embed=["a match doc", "other doc"], top_k=1,
    )
    assert out[0]["text"] == "a match doc"
    assert calls
    assert all(c[1:] == ("text-embedding-3-small", "openai") for c in calls)


def test_chroma_path_uses_embedding_provider(search_module):
    module, calls = search_module
    collection = Mock()
    collection.query.return_value = {
        "ids": [["1"]], "distances": [[0.1]], "documents": [["doc"]],
    }
    client = Mock()
    client.get_collection.return_value = collection
    module.search_similar_texts(
        "q", "text-embedding-3-small", "openai", chroma_client=client,
    )
    assert calls == [(["q"], "text-embedding-3-small", "openai")]
