"""Integration tests: npcpy.gen.decision driven by the Ollama System One backend."""

import pytest

from npcpy.gen.decision import (
    DecisionQuestion,
    DecisionRouter,
    DecisionSystem1,
    decision_choice,
    decision_noul,
    decision_predict,
    decision_route,
    decision_score,
)
from npcpy.gen.systemone import OllamaSystem1Error

from tests.ollama_mock import MockOllamaHandler


TICKET = {"ticket": "I was charged twice. Please refund the extra payment."}
QUEUE_CRITERIA = {
    "billing": "Payments and refunds",
    "technical": "Bugs and integrations",
    "other": "None of the above",
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


def test_from_ollama_decide(mock_ollama):
    model = DecisionSystem1.from_ollama(model="nimble", base_url=mock_ollama)
    assert model.backend == "ollama"
    result = model.decide(TICKET, QUESTIONS)
    assert set(result.answers) == {"team", "refund", "urgency"}
    assert result.answers["team"].choice == "billing"
    assert result.answers["urgency"].legend["2"] == "Urgent"
    assert model.to_dict()["ollama"]["model"] == "nimble"
    posts = [r for r in MockOllamaHandler.requests_seen if r["path"] == "/v1/systemone"]
    assert len(posts) == 1


def test_decision_helpers_with_ollama_backend(mock_ollama):
    chosen = decision_choice(
        TICKET,
        instructions="Which team should handle this ticket?",
        criteria=QUEUE_CRITERIA,
        backend="ollama",
        model="nimble",
        base_url=mock_ollama,
    )
    assert chosen.choice == "billing"

    noul_result = decision_noul(
        TICKET,
        instructions="Does the customer explicitly ask for a refund?",
        backend="ollama",
        base_url=mock_ollama,
    )
    assert noul_result.noul == pytest.approx(0.997)

    score_result = decision_score(
        TICKET,
        instructions="How urgent is this ticket?",
        criteria=["Routine", "Soon", "Urgent"],
        backend="ollama",
        base_url=mock_ollama,
    )
    assert score_result.score == pytest.approx(0.815)

    batch = decision_predict(TICKET, QUESTIONS, backend="ollama", base_url=mock_ollama)
    assert set(batch.answers) == {"team", "refund", "urgency"}


def test_backend_from_env(mock_ollama, monkeypatch):
    monkeypatch.setenv("NPCPY_SYSTEM1_BACKEND", "ollama")
    monkeypatch.setenv("NPCPY_SYSTEM1_OLLAMA_MODEL", "nimble")
    monkeypatch.setenv("NPCPY_OLLAMA_BASE_URL", mock_ollama)
    result = decision_choice(
        TICKET,
        instructions="Which team should handle this ticket?",
        criteria=QUEUE_CRITERIA,
    )
    assert result.choice == "billing"


def test_route_uses_ollama_directly(mock_ollama):
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
        instructions="Which team should handle this ticket?",
        tiers=router.tiers,
        criteria=router.criteria,
        backend="ollama",
        base_url=mock_ollama,
    )
    assert result["tier"] in router.tiers
    assert result["tiers"] == router.tiers
    assert isinstance(result["confidence"], float)
    assert result["probabilities"]


def test_route_threshold_falls_back_to_last_tier(mock_ollama):
    router = DecisionRouter(tiers=["tier_a", "tier_b"])
    result = decision_route(
        TICKET,
        instructions="Pick a tier.",
        tiers=router.tiers,
        criteria=router.criteria,
        threshold=0.999,  # mock confidence is 0.8906
        backend="ollama",
        base_url=mock_ollama,
    )
    assert result["tier"] == "tier_b"


def test_training_not_supported_for_ollama():
    model = DecisionSystem1.from_ollama(model="nimble")
    with pytest.raises(ValueError, match="pre-trained"):
        model.train([{"state": "x", "type": "noul", "instructions": "?", "answer": {"noul": 1.0}}])


def test_loading_local_path_not_supported_for_ollama(tmp_path):
    model = DecisionSystem1.from_ollama(model="nimble")
    with pytest.raises(ValueError, match="does not load local model paths"):
        model.load(str(tmp_path))


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown decision backend"):
        DecisionSystem1(backend="quantum")


def test_ollama_unavailable_surfaces_clear_error():
    with pytest.raises(OllamaSystem1Error, match="ollama serve"):
        decision_choice(
            TICKET,
            instructions="Which team?",
            criteria=QUEUE_CRITERIA,
            backend="ollama",
            base_url="http://127.0.0.1:59998",
        )
