"""get_mlx_response must forward top_k to mlx_lm's make_sampler."""
import sys
import types
from unittest import mock

from npcpy.gen.response import get_mlx_response


def _install_stub_mlx_lm():
    sampler_calls = []

    class _Tok:
        def apply_chat_template(self, messages, **kwargs):
            return "chat"

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.load = lambda *a, **k: (object(), _Tok())
    mlx_lm.generate = lambda *a, **k: "ok"
    sample_utils = types.ModuleType("mlx_lm.sample_utils")

    def make_sampler(**kwargs):
        sampler_calls.append(kwargs)
        return lambda x: x

    sample_utils.make_sampler = make_sampler
    mlx_lm.sample_utils = sample_utils
    return {"mlx_lm": mlx_lm, "mlx_lm.sample_utils": sample_utils}, sampler_calls


def test_mlx_forwards_top_k_to_sampler():
    modules, calls = _install_stub_mlx_lm()
    with mock.patch.dict(sys.modules, modules):
        get_mlx_response(prompt="hi", model="m", temperature=0.5, top_p=0.9, top_k=40)
    assert calls == [{"temp": 0.5, "top_p": 0.9, "top_k": 40}]


def test_mlx_omits_top_k_when_not_given():
    modules, calls = _install_stub_mlx_lm()
    with mock.patch.dict(sys.modules, modules):
        get_mlx_response(prompt="hi", model="m", temperature=0.5, top_p=0.9)
    assert calls == [{"temp": 0.5, "top_p": 0.9}]
