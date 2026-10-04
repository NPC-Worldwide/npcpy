"""Tests for the Ollama System One backend (``/v1/systemone``).

These run against a local mock Ollama server that reproduces the documented
response shape, so they need neither a real Ollama install nor the ``nimble``
weights.
"""

import pytest

from npcpy.ft.system1 import ChoiceResult, NoulResult, ScoreResult
from npcpy.gen.systemone import (
    MAX_REQUEST_BYTES,
    OllamaSystem1Client,
    OllamaSystem1Config,
    OllamaSystem1Error,
    _normalize_base_url,
    available,
    list_models,
    parse_answers,
    require_available,
    systemone,
)
from tests.ollama_mock import MockOllamaHandler


TICKET = {"ticket": "I was charged twice. Please refund the extra payment."}
QUEUE_CRITERIA = {
    "billing": "Payments and refunds",
    "technical": "Bugs and integrations",
    "other": "None of the above",
}


def test_normalize_base_url_variants():
    assert _normalize_base_url("http://localhost:11434") == "http://localhost:11434"
    assert _normalize_base_url("127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert _normalize_base_url("http://localhost:11434/v1") == "http://localhost:11434"
    assert _normalize_base_url("http://localhost:11434/v1/systemone") == "http://localhost:11434"
    assert _normalize_base_url("http://host:11434/") == "http://host:11434"
    assert _normalize_base_url("localhost") == "http://localhost:11434"


def test_choice_parses_response(mock_ollama):
    client = OllamaSystem1Client(OllamaSystem1Config(model="nimble", base_url=mock_ollama))
    result = client.choice(
        TICKET,
        instructions="Which team should handle this ticket?",
        criteria=QUEUE_CRITERIA,
    )
    assert isinstance(result, ChoiceResult)
    assert result.choice == "billing"
    assert set(result.probabilities) == set(QUEUE_CRITERIA)
    assert result.confidence == pytest.approx(0.8906)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-3)


def test_noul_and_score_parsing(mock_ollama):
    client = OllamaSystem1Client(OllamaSystem1Config(model="nimble", base_url=mock_ollama))
    noul_result = client.noul(
        TICKET, instructions="Does the customer explicitly ask for a refund?"
    )
    assert isinstance(noul_result, NoulResult)
    assert noul_result.noul == pytest.approx(0.997)
    assert 0.0 <= noul_result.noul <= 1.0

    score_result = client.score(
        TICKET,
        instructions="How urgent is this ticket?",
        criteria=["Routine", "Soon", "Urgent"],
    )
    assert isinstance(score_result, ScoreResult)
    assert score_result.score == pytest.approx(0.815)
    assert score_result.legend == {"0": "Routine", "1": "Soon", "2": "Urgent"}
    assert set(score_result.probabilities) == {"0", "1", "2"}
    assert score_result.to_dict()["legend"]["2"] == "Urgent"


def test_predict_all_questions_in_one_request(mock_ollama):
    client = OllamaSystem1Client(OllamaSystem1Config(model="nimble", base_url=mock_ollama))
    result = client.predict(
        TICKET,
        {
            "team": {"type": "choice", "instructions": "Which team?", "criteria": QUEUE_CRITERIA},
            "refund": {"type": "noul", "instructions": "Refund requested?"},
            "urgency": {"type": "score", "instructions": "Urgency?", "criteria": ["Routine", "Soon", "Urgent"]},
        },
    )
    assert set(result.answers) == {"team", "refund", "urgency"}
    assert isinstance(result.answers["team"], ChoiceResult)
    assert isinstance(result.answers["refund"], NoulResult)
    assert isinstance(result.answers["urgency"], ScoreResult)
    assert result.routing["usage"]["input_tokens"] == 174

    posts = [r for r in MockOllamaHandler.requests_seen if r["path"] == "/v1/systemone"]
    assert len(posts) == 1, "all questions must be scored in a single request"
    body = posts[0]["body"]
    assert body["model"] == "nimble"
    assert isinstance(body["state"], dict)
    assert body["questions"]["team"]["criteria"] == QUEUE_CRITERIA


def test_keep_alive_and_state_string(mock_ollama):
    client = OllamaSystem1Client(
        OllamaSystem1Config(model="nimble", base_url=mock_ollama, keep_alive="10m")
    )
    client.choice("plain text state", instructions="Which label?", criteria=QUEUE_CRITERIA)
    body = MockOllamaHandler.requests_seen[-1]["body"]
    assert body["keep_alive"] == "10m"
    assert body["state"] == "plain text state"


def test_state_text_is_not_mutated_for_lists(mock_ollama):
    systemone(
        [{"role": "user", "text": "hi"}],
        {"q": {"type": "noul", "instructions": "Is it a greeting?"}},
        base_url=mock_ollama,
    )
    body = MockOllamaHandler.requests_seen[-1]["body"]
    assert body["state"] == [{"role": "user", "text": "hi"}]


def test_question_type_aliases(mock_ollama):
    client = OllamaSystem1Client(OllamaSystem1Config(model="nimble", base_url=mock_ollama))
    result = client.predict(
        TICKET,
        {
            "a": {"type": "yes_no", "instructions": "Refund?"},
            "b": {"type": "classify", "instructions": "Team?", "criteria": QUEUE_CRITERIA},
        },
    )
    assert isinstance(result.answers["a"], NoulResult)
    assert isinstance(result.answers["b"], ChoiceResult)


def test_unknown_question_type_raises():
    with pytest.raises(ValueError):
        systemone("x", {"q": {"type": "wat", "instructions": "?"}})


def test_choice_requires_mapping_criteria():
    with pytest.raises(ValueError):
        systemone("x", {"q": {"type": "choice", "instructions": "?", "criteria": ["a", "b"]}})


def test_empty_state_raises():
    with pytest.raises(ValueError):
        systemone("", {"q": {"type": "noul", "instructions": "?"}})


def test_request_size_guard():
    big_state = "x" * (MAX_REQUEST_BYTES + 1)
    with pytest.raises(OllamaSystem1Error, match="64 KiB"):
        systemone(big_state, {"q": {"type": "noul", "instructions": "?"}})


def test_404_mentions_upgrade(mock_ollama):
    MockOllamaHandler.fail_status = 404
    MockOllamaHandler.fail_body = {"error": "page not found"}
    with pytest.raises(OllamaSystem1Error, match="0.35.0"):
        systemone("x", {"q": {"type": "noul", "instructions": "?"}}, base_url=mock_ollama)


def test_413_mentions_limit(mock_ollama):
    MockOllamaHandler.fail_status = 413
    MockOllamaHandler.fail_body = {"error": "request body must not exceed 64 KiB"}
    with pytest.raises(OllamaSystem1Error, match="64 KiB"):
        systemone("x", {"q": {"type": "noul", "instructions": "?"}}, base_url=mock_ollama)


def test_400_mentions_model(mock_ollama):
    MockOllamaHandler.fail_status = 400
    MockOllamaHandler.fail_body = {"error": "unsupported model"}
    with pytest.raises(OllamaSystem1Error, match="nimble"):
        systemone("x", {"q": {"type": "noul", "instructions": "?"}}, base_url=mock_ollama)


def test_connection_error_is_actionable():
    with pytest.raises(OllamaSystem1Error, match="ollama serve"):
        systemone(
            "x",
            {"q": {"type": "noul", "instructions": "?"}},
            base_url="http://127.0.0.1:59999",
            timeout=1,
        )


def test_systemone_allows_unbounded_request_timeout():
    seen = {}

    class Response:
        ok = True

        def json(self):
            return {"answers": {"q": {"type": "noul", "noul": 1.0}}}

    class Session:
        def post(self, url, **kwargs):
            seen.update(kwargs)
            return Response()

    systemone("state", {"q": {"type": "noul", "instructions": "Is this true?"}}, timeout=None, session=Session())
    assert seen["timeout"] is None


def test_available_reports_good_state(mock_ollama):
    info = available(model="nimble", base_url=mock_ollama)
    assert info["available"] is True
    assert info["version"] == "0.35.1"
    assert info["version_ok"] is True
    assert info["model_present"] is True
    require_available(model="nimble", base_url=mock_ollama)


def test_available_reports_old_version(mock_ollama):
    MockOllamaHandler.version = "0.24.0"
    info = available(model="nimble", base_url=mock_ollama)
    assert info["reachable"] is True
    assert info["version_ok"] is False
    assert info["available"] is False
    with pytest.raises(OllamaSystem1Error, match="does not support System One"):
        require_available(model="nimble", base_url=mock_ollama)


def test_available_reports_missing_model(mock_ollama):
    info = available(model="nimble", base_url=mock_ollama)
    assert info["model_present"] is True
    missing = available(model="tev1", base_url=mock_ollama)
    assert missing["model_present"] is False
    assert missing["available"] is False
    with pytest.raises(OllamaSystem1Error, match="ollama pull tev1"):
        require_available(model="tev1", base_url=mock_ollama)


def test_list_models(mock_ollama):
    models = list_models(base_url=mock_ollama)
    assert "nimble:latest" in models


def test_parse_answers_tolerates_missing_fields():
    payload = {
        "model": "nimble",
        "answers": {
            "team": {"type": "choice", "probabilities": {"billing": 0.9, "bug": 0.1}},
            "refund": {"type": "noul"},
            "urgency": {"type": "score", "probabilities": {"0": 0.5, "1": 0.5}},
        },
        "usage": {},
    }
    result = parse_answers(payload)
    assert result.answers["team"].choice == "billing"
    assert result.answers["team"].confidence == pytest.approx(0.9)
    assert result.answers["refund"].noul == pytest.approx(0.5)
    assert result.answers["urgency"].score == pytest.approx(0.5)


def test_score_result_legend_optional_in_to_dict():
    local = ScoreResult(score=1.0, probabilities={"0": 0.5}, confidence=0.5)
    assert "legend" not in local.to_dict()
    assert local.legend is None
