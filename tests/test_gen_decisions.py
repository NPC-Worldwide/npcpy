"""Tests for the hosted Decisions API client (npcpy.gen.decisions)."""

import json

import pytest

from npcpy.ft.system1 import ChoiceResult, NoulResult, ScoreResult, System1Result
from npcpy.gen.decisions import (
    PROVIDERS,
    DecisionsClient,
    DecisionsConfig,
    DecisionsError,
    build_payload,
    get_decision_response,
    load_decisions,
    parse_answers,
)

from tests.decisions_mock import MockDecisionsHandler


PROVIDER_NAMES = sorted(PROVIDERS)
TICKET = {"ticket": "I was charged twice. Please refund the extra payment."}
QUEUE_CRITERIA = {
    "billing": "Payments and refunds",
    "technical": "Bugs and integrations",
}
SEVERITY = ["Cosmetic", "Workaround available", "Fully blocked"]


# --------------------------------------------------------------------------- #
# payload building
# --------------------------------------------------------------------------- #

def test_payload_openai_shape():
    body = build_payload(
        "openai",
        "gpt-6-luna",
        "charged twice",
        {"dept": {"type": "choice", "instructions": "Where?", "criteria": QUEUE_CRITERIA}},
    )
    assert body["model"] == "gpt-6-luna"
    assert body["input"] == "charged twice"
    assert isinstance(body["questions"], list)
    question = body["questions"][0]
    assert question["type"] == "choice"
    assert question["name"] == "dept"
    assert [c["value"] for c in question["choices"]] == list(QUEUE_CRITERIA)


def test_payload_gemini_shape():
    body = build_payload(
        "gemini",
        "gemini-decision-1",
        "charged twice",
        {"dept": {"type": "choice", "instructions": "Where?", "criteria": QUEUE_CRITERIA}},
    )
    assert body["model"] == "gemini-decision-1"
    assert body["input"] == "charged twice"
    assert isinstance(body["questions"], dict)
    assert body["questions"]["dept"]["criteria"] == QUEUE_CRITERIA


def test_payload_perplexity_uses_state():
    body = build_payload(
        "perplexity",
        "pplx-decider-latest",
        TICKET,
        {"dept": {"type": "choice", "instructions": "Where?", "criteria": QUEUE_CRITERIA}},
    )
    assert body["state"] == TICKET
    assert isinstance(body["questions"], dict)


def test_payload_maps_noul_to_predicate_for_openai():
    body = build_payload("openai", "m", "text", {"q": {"type": "noul", "instructions": "?"}})
    assert body["questions"][0]["type"] == "predicate"


def test_payload_keeps_predicate_as_openai_native_type():
    body = build_payload("openai", "m", "text", {"q": {"type": "predicate", "instructions": "?"}})
    assert body["questions"][0]["type"] == "predicate"


def test_payload_score_levels_are_split_for_openai():
    body = build_payload(
        "openai", "m", "text", {"sev": {"type": "score", "instructions": "?", "criteria": SEVERITY}}
    )
    levels = body["questions"][0]["levels"]
    assert levels[0] == {"label": "Cosmetic", "description": "Cosmetic"}
    assert levels[1] == {"label": "Workaround available", "description": "Workaround available"}


def test_payload_openai_json_state_uses_input_json_part():
    body = build_payload("openai", "m", TICKET, {"q": {"type": "noul", "instructions": "?"}})
    assert body["input"][0]["content"][0]["type"] == "input_json"


def test_payload_gemini_json_state_is_serialized_text():
    body = build_payload("gemini", "m", TICKET, {"q": {"type": "noul", "instructions": "?"}})
    assert json.loads(body["input"]) == TICKET


def test_payload_noul_criteria_folded_into_openai_instructions():
    body = build_payload(
        "openai",
        "m",
        "text",
        {
            "q": {
                "type": "noul",
                "instructions": "Is it bad?",
                "criteria": {"true": "broken", "false": "fine"},
            }
        },
    )
    instructions = body["questions"][0]["instructions"]
    assert "Answer yes when: broken" in instructions
    assert "Answer no when: fine" in instructions


def test_payload_noul_true_false_criteria_for_gemini():
    body = build_payload(
        "gemini",
        "m",
        "text",
        {
            "q": {
                "type": "noul",
                "instructions": "?",
                "criteria": {"true": "broken", "false": "fine"},
            }
        },
    )
    assert body["questions"]["q"]["criteria"] == {
        "true_description": "broken",
        "false_description": "fine",
    }


def test_payload_rejects_empty_questions():
    with pytest.raises(DecisionsError):
        build_payload("openai", "m", "text", {})


def test_payload_rejects_unknown_question_type():
    with pytest.raises(DecisionsError):
        build_payload("openai", "m", "text", {"q": {"type": "locate", "instructions": "?"}})


def test_payload_rejects_unknown_provider():
    with pytest.raises(DecisionsError):
        build_payload("bogus", "m", "text", {"q": {"type": "noul", "instructions": "?"}})


def test_payload_rejects_missing_model():
    with pytest.raises(DecisionsError):
        build_payload("openai", "", "text", {"q": {"type": "noul", "instructions": "?"}})


def test_choice_question_requires_criteria():
    with pytest.raises(DecisionsError, match="requires criteria"):
        build_payload("openai", "m", "text", {"q": {"type": "choice", "instructions": "?"}})


def test_score_question_requires_criteria():
    with pytest.raises(DecisionsError, match="requires criteria"):
        build_payload("openai", "m", "text", {"q": {"type": "score", "instructions": "?"}})


# --------------------------------------------------------------------------- #
# response parsing
# --------------------------------------------------------------------------- #

def test_parse_answers_map_shape():
    payload = {
        "model": "m",
        "answers": {
            "yes": {"type": "noul", "noul": 0.8},
            "pick": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.9,
                "probabilities": {"billing": 0.9, "technical": 0.1},
            },
            "sev": {
                "type": "score",
                "score": 1.5,
                "confidence": 0.4,
                "legend": {"0": "low", "1": "mid", "2": "high"},
                "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5},
            },
        },
    }
    result = parse_answers(payload, provider="gemini")
    assert isinstance(result, System1Result)
    assert isinstance(result.answers["yes"], NoulResult)
    assert result.answers["yes"].noul == pytest.approx(0.8)
    assert isinstance(result.answers["pick"], ChoiceResult)
    assert result.answers["pick"].choice == "billing"
    assert isinstance(result.answers["sev"], ScoreResult)
    assert result.answers["sev"].legend == {"0": "low", "1": "mid", "2": "high"}
    assert result.routing["provider"] == "gemini"


def test_parse_answers_list_shape_with_probability_records():
    payload = {
        "answers": [
            {"type": "predicate", "name": "dmg", "probability": 0.42},
            {
                "type": "choice",
                "name": "dept",
                "choice": "billing",
                "confidence": 0.7,
                "probabilities": [
                    {"value": "billing", "probability": 0.7},
                    {"value": "technical", "probability": 0.3},
                ],
            },
            {
                "type": "score",
                "name": "sev",
                "score": 1.1,
                "confidence": 0.55,
                "probabilities": [
                    {"value": 0, "label": "Cosmetic", "probability": 0.1},
                    {"value": 1, "label": "Workaround", "probability": 0.7},
                    {"value": 2, "label": "Blocked", "probability": 0.2},
                ],
            },
        ]
    }
    result = parse_answers(payload, provider="openai")
    assert result.answers["dmg"].noul == pytest.approx(0.42)
    assert result.answers["dept"].probabilities == {"billing": 0.7, "technical": 0.3}
    assert result.answers["sev"].legend["2"] == "Blocked"
    assert result.answers["sev"].probabilities == {"0": 0.1, "1": 0.7, "2": 0.2}


def test_parse_answers_raises_on_refusal():
    payload = {"answers": [{"type": "refusal", "name": "q", "refusal": "policy"}]}
    with pytest.raises(DecisionsError, match="refused"):
        parse_answers(payload)


def test_parse_answers_rejects_missing_answers():
    with pytest.raises(DecisionsError):
        parse_answers({"model": "m"})


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

def test_config_auth_headers_per_provider():
    assert DecisionsConfig(provider="openai", model="m", api_key="sk").auth_headers()["Authorization"] == "Bearer sk"
    assert DecisionsConfig(provider="gemini", model="m", api_key="g").auth_headers()["x-goog-api-key"] == "g"
    assert DecisionsConfig(provider="perplexity", model="m", api_key="p").auth_headers()["Authorization"] == "Bearer p"


def test_config_requires_api_key(monkeypatch):
    for info in PROVIDERS.values():
        monkeypatch.delenv(info["api_key_env"], raising=False)
    with pytest.raises(DecisionsError, match="No API key"):
        DecisionsConfig(provider="openai", model="m").auth_headers()


def test_config_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert DecisionsConfig(provider="gemini", model="m").resolved_api_key() == "from-env"


def test_config_endpoints():
    assert DecisionsConfig(provider="openai", model="m").endpoint() == "https://api.openai.com/v1/decisions"
    assert (
        DecisionsConfig(provider="gemini", model="m").endpoint()
        == "https://generativelanguage.googleapis.com/v1beta/decisions"
    )
    assert (
        DecisionsConfig(provider="perplexity", model="m").endpoint()
        == "https://api.perplexity.ai/v1/decisions"
    )


def test_config_requires_model():
    with pytest.raises(DecisionsError, match="model is required"):
        DecisionsConfig(provider="openai", model="")


def test_config_rejects_unknown_provider():
    with pytest.raises(DecisionsError, match="Unknown Decisions API provider"):
        DecisionsConfig(provider="bogus", model="m")


def test_client_defaults_api_url_to_provider():
    client = DecisionsClient(provider="gemini", model="gemini-decision-1")
    assert client.config.resolved_api_url() == PROVIDERS["gemini"]["api_url"]


# --------------------------------------------------------------------------- #
# client against the mock server
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("provider", PROVIDER_NAMES)
def test_client_choice_noul_score(provider, mock_decisions):
    client = DecisionsClient(
        provider=provider, model="test-model", api_key="test-key", api_url=mock_decisions
    )
    assert client.noul("text", instructions="Did it break?").noul > 0.5
    assert client.choice("text", instructions="Where?", criteria=QUEUE_CRITERIA).choice == "billing"
    severity = client.score("text", instructions="How bad?", criteria=SEVERITY)
    assert severity.score > 0
    assert severity.legend is not None


@pytest.mark.parametrize("provider", PROVIDER_NAMES)
def test_client_predict_multi_question(provider, mock_decisions):
    client = DecisionsClient(
        provider=provider, model="test-model", api_key="test-key", api_url=mock_decisions
    )
    result = client.predict(
        TICKET,
        {
            "refund": {"type": "noul", "instructions": "Refund requested?"},
            "dept": {"type": "choice", "instructions": "Where?", "criteria": QUEUE_CRITERIA},
            "sev": {"type": "score", "instructions": "How bad?", "criteria": SEVERITY},
        },
    )
    assert set(result.answers) == {"refund", "dept", "sev"}
    assert isinstance(result.answers["refund"], NoulResult)
    assert isinstance(result.answers["dept"], ChoiceResult)
    assert isinstance(result.answers["sev"], ScoreResult)


@pytest.mark.parametrize("provider", PROVIDER_NAMES)
def test_client_posts_to_decisions_path(provider, mock_decisions):
    client = DecisionsClient(
        provider=provider, model="test-model", api_key="test-key", api_url=mock_decisions
    )
    client.noul("text", instructions="?")
    posts = [r for r in MockDecisionsHandler.requests_seen if r["path"].endswith("/decisions")]
    assert len(posts) == 1


def test_client_uses_provider_request_shape(mock_decisions):
    DecisionsClient(
        provider="openai", model="m", api_key="k", api_url=mock_decisions
    ).noul("text", instructions="?")
    openai_body = MockDecisionsHandler.requests_seen[-1]["body"]
    assert isinstance(openai_body["questions"], list)
    assert "input" in openai_body

    DecisionsClient(
        provider="perplexity", model="m", api_key="k", api_url=mock_decisions
    ).noul("text", instructions="?")
    pplx_body = MockDecisionsHandler.requests_seen[-1]["body"]
    assert "state" in pplx_body
    assert isinstance(pplx_body["questions"], dict)


def test_client_sends_configured_model(mock_decisions):
    DecisionsClient(
        provider="gemini", model="my-decision-model", api_key="k", api_url=mock_decisions
    ).noul("text", instructions="?")
    assert MockDecisionsHandler.requests_seen[-1]["body"]["model"] == "my-decision-model"


def test_client_refusal_raises(mock_decisions):
    MockDecisionsHandler.refusal_question = "q"
    client = DecisionsClient(
        provider="openai", model="m", api_key="k", api_url=mock_decisions
    )
    with pytest.raises(DecisionsError, match="refused"):
        client.noul("text", instructions="?", question_name="q")


def test_client_error_includes_provider_hint(mock_decisions):
    MockDecisionsHandler.fail_status = 401
    MockDecisionsHandler.fail_body = {"error": {"message": "bad key"}}
    client = DecisionsClient(
        provider="gemini", model="m", api_key="k", api_url=mock_decisions
    )
    with pytest.raises(DecisionsError, match="rejected the API key"):
        client.noul("text", instructions="?")


def test_client_400_mentions_provider_request_shape(mock_decisions):
    MockDecisionsHandler.fail_status = 400
    MockDecisionsHandler.fail_body = {"error": {"message": "bad question"}}
    with pytest.raises(DecisionsError, match="rejected the decisions request"):
        get_decision_response(
            "text",
            {"q": {"type": "noul", "instructions": "?"}},
            provider="openai",
            model="m",
            api_key="k",
            api_url=mock_decisions,
        )


def test_decisions_returns_raw_payload(mock_decisions):
    payload = get_decision_response(
        "text",
        {"q": {"type": "noul", "instructions": "?"}},
        provider="gemini",
        model="gemini-decision-1",
        api_key="k",
        api_url=mock_decisions,
    )
    assert "answers" in payload
    assert payload["model"] == "gemini-decision-1"


def test_load_decisions_returns_client(mock_decisions):
    client = load_decisions(
        provider="openai", model="m", api_key="k", api_url=mock_decisions
    )
    assert isinstance(client, DecisionsClient)
    assert client.to_dict()["provider"] == "openai"
