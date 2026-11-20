"""Tests: npcpy.gen.decision driven by a hosted Decisions API provider."""

import pytest

from npcpy.gen.decision import (
    DecisionQuestion,
    DecisionRouter,
    DecisionSystem1,
    _resolve_provider,
    decision_choice,
    decision_noul,
    decision_predict,
    decision_route,
    decision_score,
)
from npcpy.gen.decisions import DecisionsError


TICKET = {"ticket": "I was charged twice. Please refund the extra payment."}
QUEUE_CRITERIA = {
    "billing": "Payments and refunds",
    "technical": "Bugs and integrations",
}
QUESTIONS = [
    DecisionQuestion(
        name="team",
        type="choice",
        instructions="Which team should handle this ticket?",
        criteria=QUEUE_CRITERIA,
    ),
    DecisionQuestion(
        name="refund",
        type="noul",
        instructions="Does the customer explicitly ask for a refund?",
    ),
    DecisionQuestion(
        name="urgency",
        type="score",
        instructions="How urgent is this ticket?",
        criteria=["Routine", "Soon", "Urgent"],
    ),
]


# --------------------------------------------------------------------------- #
# provider resolution -- same vocabulary as get_llm_response
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("ollama", "ollama"),
        ("OLLAMA", "ollama"),
        ("openai", "openai"),
        ("gemini", "gemini"),
        ("perplexity", "perplexity"),
        ("PERPLEXITY", "perplexity"),
    ],
)
def test_resolve_provider(raw, expected):
    assert _resolve_provider(raw) == expected


@pytest.mark.parametrize("raw", ["local", "sklearn", "train", "bogus"])
def test_resolve_provider_rejects_non_providers(raw):
    """Only real named providers are accepted; there is no 'local' provider."""
    with pytest.raises(ValueError, match="Unknown decision provider"):
        _resolve_provider(raw)


def test_no_provider_means_local_classifier():
    assert _resolve_provider() is None


def test_backend_is_a_deprecated_alias():
    assert _resolve_provider(backend="ollama") == "ollama"
    # explicit provider wins over backend
    assert _resolve_provider("gemini", backend="ollama") == "gemini"


# --------------------------------------------------------------------------- #
# DecisionSystem1
# --------------------------------------------------------------------------- #

def test_provider_builds_client():
    model = DecisionSystem1(provider="gemini", model="gemini-decision-1", api_key="k")
    assert model.provider == "gemini"
    assert model.predictor.provider == "gemini"
    assert model.to_dict()["model"] == "gemini-decision-1"


def test_from_hosted_builds_client():
    model = DecisionSystem1.from_hosted(
        provider="gemini", model="gemini-decision-1", api_key="k"
    )
    assert model.provider == "gemini"
    assert model.to_dict()["provider"] == "gemini"


def test_hosted_requires_model():
    with pytest.raises(ValueError, match="requires model"):
        DecisionSystem1(provider="gemini")


def test_ollama_accepts_model_and_api_url():
    model = DecisionSystem1(provider="ollama", model="tev1:0.8b", api_url="http://localhost:11434")
    assert model.provider == "ollama"
    assert model.to_dict()["model"] == "tev1:0.8b"
    assert model.to_dict()["api_url"] == "http://localhost:11434"


def test_hosted_train_and_load_are_rejected(tmp_path):
    model = DecisionSystem1(provider="openai", model="gpt-6-luna", api_key="k")
    with pytest.raises(ValueError, match="pre-trained"):
        model.train([])
    with pytest.raises(ValueError, match="do not load local model paths"):
        model.load(str(tmp_path / "model"))


def test_decide(mock_decisions):
    model = DecisionSystem1.from_hosted(
        provider="gemini", model="gemini-decision-1", api_key="k", api_url=mock_decisions
    )
    result = model.decide(TICKET, QUESTIONS)
    assert set(result.answers) == {"team", "refund", "urgency"}
    assert result.answers["team"].choice == "billing"
    assert result.answers["urgency"].legend["2"] == "Urgent"


# --------------------------------------------------------------------------- #
# module-level helpers
# --------------------------------------------------------------------------- #

def test_helpers_take_provider(mock_decisions, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    kwargs = dict(provider="gemini", model="gemini-decision-1", api_url=mock_decisions)
    assert (
        decision_choice(TICKET, instructions="Which team?", criteria=QUEUE_CRITERIA, **kwargs).choice
        == "billing"
    )
    assert decision_noul(TICKET, instructions="Refund requested?", **kwargs).noul > 0.5
    assert (
        decision_score(
            TICKET, instructions="How urgent?", criteria=["Routine", "Soon", "Urgent"], **kwargs
        ).score
        > 0
    )
    batch = decision_predict(TICKET, QUESTIONS, **kwargs)
    assert set(batch.answers) == {"team", "refund", "urgency"}


def test_helpers_accept_api_key(mock_decisions):
    result = decision_noul(
        TICKET,
        instructions="Refund requested?",
        provider="gemini",
        model="gemini-decision-1",
        api_key="k",
        api_url=mock_decisions,
    )
    assert result.noul > 0.5


def test_route_uses_provider_directly(mock_decisions, monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "k")
    router = DecisionRouter(
        tiers=["self_service", "agent", "engineering"],
        criteria={
            "self_service": "user can resolve with docs",
            "agent": "needs a human",
            "engineering": "requires a code fix",
        },
    )
    result = decision_route(
        TICKET,
        instructions="Which team?",
        tiers=router.tiers,
        criteria=router.criteria,
        provider="perplexity",
        model="pplx-decider-latest",
        api_url=mock_decisions,
    )
    assert result["tier"] in router.tiers
    assert 0.0 <= result["confidence"] <= 1.0


def test_route_threshold_falls_back_to_last_tier(mock_decisions, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    result = decision_route(
        TICKET,
        instructions="Which team?",
        tiers=["billing", "technical"],
        criteria={"billing": "payments", "technical": "bugs"},
        threshold=0.999,
        provider="gemini",
        model="gemini-decision-1",
        api_url=mock_decisions,
    )
    assert result["tier"] == "technical"


def test_missing_api_key_raises(monkeypatch):
    for env_name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "PERPLEXITY_API_KEY"):
        monkeypatch.delenv(env_name, raising=False)
    with pytest.raises(DecisionsError, match="No API key"):
        decision_choice(
            TICKET,
            instructions="Which team?",
            criteria=QUEUE_CRITERIA,
            provider="openai",
            model="gpt-6-luna",
        )


def test_helper_requires_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with pytest.raises(ValueError, match="requires model"):
        decision_noul(TICKET, instructions="?", provider="gemini")
