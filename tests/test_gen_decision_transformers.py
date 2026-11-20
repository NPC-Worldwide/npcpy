"""Tests for the local transformers zero-shot decisions backend."""

import pytest

from npcpy.gen.decision import (
    DecisionSystem1,
    decision_choice,
    decision_noul,
    decision_score,
    _resolve_provider,
)
from npcpy.gen.decisions import (
    DEFAULT_TRANSFORMERS_MODEL,
    TRANSFORMERS_PROVIDER,
    DecisionsError,
    TransformersDecisionsClient,
    load_transformers_decisions,
)


class FakePipeline:
    """Stand-in for transformers' zero-shot-classification pipeline."""

    calls = []

    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, text, candidate_labels=None, **kwargs):
        FakePipeline.calls.append({"text": text, "labels": list(candidate_labels or [])})
        scores = []
        for label in candidate_labels:
            scores.append(float(self.mapping.get(label, 0.0)))
        total = sum(scores) or 1.0
        scores = [s / total for s in scores]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return {
            "sequence": text,
            "labels": [candidate_labels[i] for i in order],
            "scores": [scores[i] for i in order],
        }


@pytest.fixture
def fake_tf(monkeypatch):
    FakePipeline.calls = []
    mapping = {"billing": 9.0, "technical": 1.0, "yes": 8.0, "no": 2.0,
               "Routine": 1.0, "Soon": 3.0, "Urgent": 6.0}
    client = TransformersDecisionsClient(model="fake-model")
    client._pipeline = FakePipeline(mapping)
    return client


def test_transformers_is_a_named_provider():
    assert _resolve_provider("transformers") == TRANSFORMERS_PROVIDER


def test_choice(fake_tf):
    result = fake_tf.choice(
        "I was charged twice.",
        instructions="Which team?",
        criteria={"billing": "Payments", "technical": "Bugs"},
    )
    assert result.choice == "billing"
    assert result.probabilities["billing"] == pytest.approx(0.9)
    assert result.probabilities["technical"] == pytest.approx(0.1)
    assert result.confidence == pytest.approx(0.9)


def test_noul(fake_tf):
    result = fake_tf.noul("charged twice", instructions="Refund requested?")
    assert result.noul == pytest.approx(0.8)


def test_score_is_probability_weighted_average(fake_tf):
    result = fake_tf.score(
        "export fails",
        instructions="How severe?",
        criteria=["Routine", "Soon", "Urgent"],
    )
    # 0*0.1 + 1*0.3 + 2*0.6 = 1.5
    assert result.score == pytest.approx(1.5)
    assert result.legend == {"0": "Routine", "1": "Soon", "2": "Urgent"}
    assert result.probabilities["2"] == pytest.approx(0.6)


def test_predict_batches_questions(fake_tf):
    result = fake_tf.predict(
        "charged twice",
        {
            "team": {"type": "choice", "instructions": "Which team?",
                     "criteria": {"billing": "Payments", "technical": "Bugs"}},
            "refund": {"type": "noul", "instructions": "Refund?"},
            "sev": {"type": "score", "instructions": "How bad?",
                    "criteria": ["Routine", "Soon", "Urgent"]},
        },
    )
    assert set(result.answers) == {"team", "refund", "sev"}
    assert result.routing["provider"] == TRANSFORMERS_PROVIDER


def test_pipeline_receives_state_and_instructions(fake_tf):
    fake_tf.choice("STATE-TEXT", instructions="WHICH-TEAM",
                   criteria={"billing": "P", "technical": "B"})
    call = FakePipeline.calls[-1]
    assert "STATE-TEXT" in call["text"]
    assert "WHICH-TEAM" in call["text"]
    assert call["labels"] == ["billing", "technical"]


def test_decision_system1_dispatch(fake_tf, monkeypatch):
    monkeypatch.setattr(
        "npcpy.gen.decision.load_transformers_decisions",
        lambda model=None, **kw: fake_tf,
    )
    model = DecisionSystem1(provider="transformers", model="fake-model")
    assert model.provider == TRANSFORMERS_PROVIDER
    assert model.to_dict()["model"] == "fake-model"
    assert model.noul("charged twice", instructions="Refund?").noul == pytest.approx(0.8)


def test_decision_system1_defaults_model(monkeypatch):
    monkeypatch.setattr(
        "npcpy.gen.decision.load_transformers_decisions",
        lambda model=None, **kw: TransformersDecisionsClient(model=model),
    )
    model = DecisionSystem1(provider="transformers")
    assert model.to_dict()["model"] == DEFAULT_TRANSFORMERS_MODEL


def test_helpers_with_transformers(fake_tf, monkeypatch):
    monkeypatch.setattr(
        "npcpy.gen.decision.load_transformers_decisions",
        lambda model=None, **kw: fake_tf,
    )
    assert decision_choice(
        "charged twice", instructions="Which team?",
        criteria={"billing": "P", "technical": "B"}, provider="transformers",
    ).choice == "billing"
    assert decision_noul(
        "charged twice", instructions="Refund?", provider="transformers"
    ).noul == pytest.approx(0.8)
    assert decision_score(
        "export fails", instructions="How bad?",
        criteria=["Routine", "Soon", "Urgent"], provider="transformers",
    ).score == pytest.approx(1.5)


def test_missing_transformers_gives_actionable_error(monkeypatch):
    client = TransformersDecisionsClient(model="some-model")
    monkeypatch.setitem(__import__("sys").modules, "transformers", None)
    with pytest.raises(DecisionsError, match="pip install transformers torch"):
        client.noul("text", instructions="?")


def test_load_transformers_decisions():
    client = load_transformers_decisions(model="m")
    assert isinstance(client, TransformersDecisionsClient)
    assert client.to_dict() == {"provider": TRANSFORMERS_PROVIDER, "model": "m"}


def test_transformers_rejects_empty_model():
    with pytest.raises(DecisionsError, match="model is required"):
        TransformersDecisionsClient(model="")
