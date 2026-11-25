import sys
from unittest.mock import MagicMock

import pytest

from npcpy.gen.embeddings import get_ollama_embeddings


def test_image_path_raises_instead_of_embedding_path_text(tmp_path, monkeypatch):
    image = tmp_path / "x.png"
    image.write_bytes(b"not read")
    fake_ollama = MagicMock()
    fake_ollama.embed.return_value = {"embeddings": [[0.1]]}
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    with pytest.raises(ValueError, match="nomic-embed-text"):
        get_ollama_embeddings([str(image)])

    fake_ollama.embed.assert_not_called()
    fake_ollama.embeddings.assert_not_called()


def test_text_inputs_are_still_embedded(monkeypatch):
    fake_ollama = MagicMock()
    fake_ollama.embeddings.return_value = {"embedding": [0.1, 0.2]}
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    assert get_ollama_embeddings(["hello"]) == [[0.1, 0.2]]
