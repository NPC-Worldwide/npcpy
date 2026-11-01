from types import SimpleNamespace

from npcpy.gen.response import provider_cost_from_usage


def test_cost_from_object_attribute():
    assert provider_cost_from_usage(SimpleNamespace(cost=0.00123)) == 0.00123


def test_cost_from_dict_key():
    assert provider_cost_from_usage({"cost": 0.00456}) == 0.00456


def test_missing_cost_returns_none():
    assert provider_cost_from_usage({}) is None
    assert provider_cost_from_usage(SimpleNamespace()) is None
    assert provider_cost_from_usage(None) is None


def test_invalid_cost_returns_none():
    assert provider_cost_from_usage({"cost": "abc"}) is None
    assert provider_cost_from_usage({"cost": [1]}) is None


def test_zero_cost_preserved():
    assert provider_cost_from_usage({"cost": 0.0}) == 0.0


def test_numeric_string_coerced():
    assert provider_cost_from_usage({"cost": "0.5"}) == 0.5
