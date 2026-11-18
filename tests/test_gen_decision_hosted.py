"""Tests: npcpy.gen.decision driven by the hosted Decisions API backend."""

import pytest

from npcpy.gen.decision import (
    DecisionQuestion,
    DecisionRouter,
    DecisionSystem1,
    _resolve_backend,
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


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("hosted", "hosted"),
        ("api", "hosted"),
        ("decisions", "hosted"),
        ("cloud", "hosted"),
        ("gemini", "hosted:gemini"),
        ("google", "hosted:gemini"),
        ("pplx", "hosted:perplexity"),
        ("hosted:gemini", "hosted:gemini"),
        ("hosted_perplexity", "hosted:perplexity"),
        ("local", "local"),
        ("ollama", "ollama"),
    ],
)
def test_resolve_backend_hosted(raw, expected):
    assert _resolve_backend(raw) == expected


def test_resolve_backend_rejects_unknown():
    with pytest.raises(ValueError):
        _resolve_backend("bogus")


def test_from_hosted_builds_client():
    model = DecisionSystem1.from_hosted(provider="gemini", api_key="k")
    assert model.backend == "hosted"
    assert model.to_dict()["hosted"]["provider"] == "gemini"
    assert model.predictor.provider == "gemini"


def test_from_hosted_explicit_provider_in_backend():
    model = DecisionSystem1(backend="hosted:perplexity", api_key="k")
    assert model.to_dict()["hosted"]["provider"] == "perplexity"


def test_hosted_train_and_load_are_rejected(tmp_path):
    model = DecisionSystem1.from_hosted(provider="openai", api_key="k")
    with pytest.raises(ValueError, match="pre-trained"):
        model.train([])
    with pytest.raises(ValueError, match="does not load local model paths"):
        model.load(str(tmp_path / "model"))


def test_from_hosted_decide(mock_decisions):
    model = DecisionSystem1.from_hosted(
        provider="gemini", api_key="k", base_url=mock_decisions
    )
    result = model.decide(TICKET, QUESTIONS)
    assert set(result.answers) == {"team", "refund", "urgency"}
    assert result.answers["team"].choice == "billing"
    assert result.answers["urgency"].legend["2"] == "Urgent"


def test_decision_helpers_with_hosted_backend(mock_decisions, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert (
        decision_choice(
            TICKET,
            instructions="Which team?",
            criteria=QUEUE_CRITERIA,
            backend="gemini",
            base_url=mock_decisions,
        ).choice
        == "billing"
    )
    assert (
        decision_noul(
            TICKET,
            instructions="Refund requested?",
            backend="gemini",
            base_url=mock_decisions,
        ).noul
        > 0.5
    )
    assert (
        decision_score(
            TICKET,
            instructions="How urgent?",
            criteria=["Routine", "Soon", "Urgent"],
            backend="gemini",
            base_url=mock_decisions,
        ).score
        > 0
    )
    batch = decision_predict(TICKET, QUESTIONS, backend="gemini", base_url=mock_decisions)
    assert set(batch.answers) == {"team", "refund", "urgency"}


def test_backend_from_env_for_hosted(mock_decisions, monkeypatch):
    monkeypatch.setenv("NPCPY_SYSTEM1_BACKEND", "hosted")
    monkeypatch.setenv("NPCPY_DECISIONS_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("NPCPY_DECISIONS_OPENAI_BASE_URL", mock_decisions)
    result = decision_choice(
        TICKET,
        instructions="Which team?",
        criteria=QUEUE_CRITERIA,
    )
    assert result.choice == "billing"


def test_route_uses_hosted_directly(mock_decisions, monkeypatch):
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
        backend="perplexity",
        base_url=mock_decisions,
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
        backend="gemini",
        base_url=mock_decisions,
    )
    assert result["tier"] == "technical"


def test_hosted_missing_api_key_raises(monkeypatch):
    for env_name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "PERPLEXITY_API_KEY"):
        monkeypatch.delenv(env_name, raising=False)
    with pytest.raises(DecisionsError, match="No API key"):
        decision_choice(
            TICKET,
            instructions="Which team?",
            criteria=QUEUE_CRITERIA,
            backend="openai",
        )
