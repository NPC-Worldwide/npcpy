import base64
import io
import sys
import types

import pytest
from PIL import Image

from npcpy.gen import image_gen

# gpt-image-1 shut down on 2026-10-23 and dall-e-2/3 on 2026-05-12.
RETIRED = {"gpt-image-1", "dall-e-2", "dall-e-3"}


@pytest.fixture
def sent_models(monkeypatch):
    sent = []
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    def respond(**kwargs):
        sent.append(kwargs["model"])
        item = types.SimpleNamespace(b64_json=b64, url=None)
        return types.SimpleNamespace(data=[item])

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.images = types.SimpleNamespace(edit=respond, generate=respond)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    return sent


def test_generate_image_openai_default_model_is_not_retired(sent_models):
    image_gen.generate_image("a cat", model=None, provider="openai")
    assert sent_models and sent_models[0] not in RETIRED


def test_edit_image_openai_default_model_is_not_retired(sent_models, tmp_path):
    path = tmp_path / "in.png"
    Image.new("RGB", (4, 4)).save(path)
    image_gen.edit_image("make it blue", str(path), provider="openai")
    assert sent_models and sent_models[0] not in RETIRED


def test_openai_image_gen_default_model_is_not_retired(sent_models):
    image_gen.openai_image_gen("a cat")
    assert sent_models and sent_models[0] not in RETIRED
