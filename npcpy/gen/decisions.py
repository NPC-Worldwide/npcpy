"""Decisions APIs: fast typed decisions from hosted decision models.

This is the hosted counterpart to :mod:`npcpy.gen.systemone` (which speaks to a
local Ollama decision model). Three hosted providers expose the same idea -- send
shared evidence plus a set of named questions, get back typed answers with
probabilities instead of generated text -- but each spells the request a little
differently. This module normalizes all of them onto the same surface used by
:mod:`npcpy.ft.system1` and :mod:`npcpy.gen.decision`:

* question types are always ``noul`` / ``choice`` / ``score``
  (``predicate`` is accepted as an alias for ``noul``),
* results are always :class:`~npcpy.ft.system1.ChoiceResult`,
  :class:`~npcpy.ft.system1.NoulResult`, or
  :class:`~npcpy.ft.system1.ScoreResult`,
* every client exposes ``choice`` / ``noul`` / ``score`` / ``predict``.

Supported providers
-------------------
=============  ===========================  ==========================
provider       model                        request shape
=============  ===========================  ==========================
``openai``     ``gpt-6-luna``               ``input`` + ``questions`` list
``gemini``     ``gemini-decision-1``        ``input`` + ``questions`` map
``perplexity`` ``pplx-decider-latest``      ``state`` + ``questions`` map
=============  ===========================  ==========================

No provider SDK is required: each call is a single JSON ``POST`` made with
``requests``, matching how :mod:`npcpy.gen.systemone` talks to Ollama.

Example
-------
>>> from npcpy.gen.decisions import DecisionsClient
>>> client = DecisionsClient(provider="gemini")
>>> client.choice(
...     "I was charged twice for my order.",
...     instructions="Which department should handle this?",
...     criteria={"billing": "Payments and refunds.", "technical": "Bugs."},
... ).choice
'billing'
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import requests

from npcpy.ft.system1 import (
    ChoiceResult,
    NoulResult,
    ScoreResult,
    System1Result,
)

PROVIDERS = ("openai", "gemini", "perplexity")

DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-6-luna",
    "gemini": "gemini-decision-1",
    "perplexity": "pplx-decider-latest",
}

DEFAULT_BASE_URLS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "perplexity": "https://api.perplexity.ai/v1",
}

DEFAULT_TIMEOUT = 60

_API_KEY_ENV: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
}

_MODEL_ENV: Dict[str, str] = {
    "openai": "NPCPY_DECISIONS_OPENAI_MODEL",
    "gemini": "NPCPY_DECISIONS_GEMINI_MODEL",
    "perplexity": "NPCPY_DECISIONS_PERPLEXITY_MODEL",
}

_BASE_URL_ENV: Dict[str, str] = {
    "openai": "NPCPY_DECISIONS_OPENAI_BASE_URL",
    "gemini": "NPCPY_DECISIONS_GEMINI_BASE_URL",
    "perplexity": "NPCPY_DECISIONS_PERPLEXITY_BASE_URL",
}

PROVIDER_ALIASES: Dict[str, str] = {
    "openai": "openai",
    "oai": "openai",
    "gpt": "openai",
    "gemini": "gemini",
    "google": "gemini",
    "googleai": "gemini",
    "perplexity": "perplexity",
    "pplx": "perplexity",
}

_QUESTION_TYPE_ALIASES: Dict[str, str] = {
    "noul": "noul",
    "yesno": "noul",
    "yes_no": "noul",
    "yes/no": "noul",
    "bool": "noul",
    "boolean": "noul",
    "predicate": "noul",
    "classify": "choice",
    "classification": "choice",
    "choice": "choice",
    "label": "choice",
    "score": "score",
    "scoring": "score",
    "rubric": "score",
    "rating": "score",
}


class DecisionsError(RuntimeError):
    """Raised when a hosted Decisions API request fails or is misconfigured."""


# --------------------------------------------------------------------------- #
# normalization helpers
# --------------------------------------------------------------------------- #

def resolve_provider(provider: Optional[str] = None) -> str:
    """Resolve a provider name, falling back to ``NPCPY_DECISIONS_PROVIDER``."""
    raw = provider or os.environ.get("NPCPY_DECISIONS_PROVIDER") or "openai"
    key = str(raw).strip().lower()
    if key not in PROVIDER_ALIASES:
        raise DecisionsError(
            f"Unknown Decisions API provider: {provider!r}. Expected one of: {', '.join(PROVIDERS)}."
        )
    return PROVIDER_ALIASES[key]


def _normalize_question_type(question_type: Optional[str]) -> str:
    key = str(question_type or "choice").strip().lower()
    if key not in _QUESTION_TYPE_ALIASES:
        raise DecisionsError(
            f"Unknown question type: {question_type!r}. Expected one of: noul, choice, score."
        )
    return _QUESTION_TYPE_ALIASES[key]


def _state_to_text(state: Union[str, Dict[str, Any], List[Any]]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, default=str)


def _normalize_questions(
    questions: Union[Dict[str, Any], List[Any]],
) -> Dict[str, Dict[str, Any]]:
    """Coerce questions into the canonical ``{name: {type, instructions, criteria}}`` shape."""
    if isinstance(questions, list):
        normalized: Dict[str, Dict[str, Any]] = {}
        for index, question in enumerate(questions):
            if isinstance(question, dict):
                name = question.get("name") or f"question_{index}"
            else:
                name = getattr(question, "name", None) or f"question_{index}"
            normalized[str(name)] = _normalize_question(question)
        if not normalized:
            raise DecisionsError("questions must contain at least one question")
        return normalized
    if not isinstance(questions, dict):
        raise DecisionsError("questions must be a mapping of name -> question or a list")
    out: Dict[str, Dict[str, Any]] = {}
    for name, question in questions.items():
        out[str(name)] = _normalize_question(question)
    if not out:
        raise DecisionsError("questions must contain at least one question")
    return out


def _normalize_question(question: Any) -> Dict[str, Any]:
    if isinstance(question, dict):
        qtype = question.get("type", "choice")
        instructions = question.get("instructions", "")
        criteria = question.get("criteria")
        if criteria is None:
            criteria = question.get("choices")
        if criteria is None:
            criteria = question.get("levels")
    else:
        qtype = getattr(question, "type", "choice")
        instructions = getattr(question, "instructions", "")
        criteria = getattr(question, "criteria", None)
    qtype = _normalize_question_type(qtype)
    payload: Dict[str, Any] = {"type": qtype, "instructions": instructions or ""}
    if criteria is not None:
        if qtype == "choice" and not isinstance(criteria, (dict, list)):
            raise DecisionsError(
                "choice questions require a criteria mapping of label -> description"
            )
        if qtype == "score" and not isinstance(criteria, list):
            raise DecisionsError("score questions require an ordered list of criteria levels")
        payload["criteria"] = _normalize_criteria(qtype, criteria)
    return payload


def _normalize_criteria(qtype: str, criteria: Any) -> Any:
    """Flatten provider-native criteria spellings into our canonical form."""
    if qtype == "choice" and isinstance(criteria, list):
        mapping = {}
        for entry in criteria:
            if isinstance(entry, dict):
                value = entry.get("value", entry.get("label"))
                if value is None:
                    continue
                mapping[str(value)] = entry.get("description") or entry.get("label") or ""
            else:
                mapping[str(entry)] = ""
        return mapping
    if qtype == "score" and isinstance(criteria, list):
        levels = []
        for entry in criteria:
            if not isinstance(entry, dict):
                levels.append(entry)
                continue
            label = str(entry.get("label") or entry.get("description") or "")
            description = entry.get("description")
            if description and str(description) != label:
                levels.append(f"{label}: {description}")
            else:
                levels.append(label)
        return levels
    if qtype == "noul" and isinstance(criteria, dict):
        # Accept both our ``true``/``false`` spelling and the
        # ``true_description``/``false_description`` spelling.
        if "true_description" in criteria or "false_description" in criteria:
            return {
                "true": criteria.get("true_description", criteria.get("true", "")),
                "false": criteria.get("false_description", criteria.get("false", "")),
            }
    return criteria


# --------------------------------------------------------------------------- #
# provider-specific request building
# --------------------------------------------------------------------------- #

def _openai_input(state: Union[str, Dict[str, Any], List[Any]]) -> Any:
    """OpenAI takes a string or an ordered array of message items."""
    if isinstance(state, str):
        return state
    if isinstance(state, list) and state and all(
        isinstance(item, dict) and "role" in item for item in state
    ):
        return state
    return [{"role": "user", "content": [{"type": "input_json", "data": state}]}]


def _gemini_input(state: Union[str, Dict[str, Any], List[Any]]) -> Any:
    """Gemini takes a string or a list of content blocks; JSON must be text."""
    if isinstance(state, str):
        return state
    if isinstance(state, list) and state and all(
        isinstance(item, dict) and "type" in item for item in state
    ):
        return state
    return _state_to_text(state)


def _openai_question(name: str, question: Dict[str, Any]) -> Dict[str, Any]:
    qtype = question["type"]
    instructions = question["instructions"]
    criteria = question.get("criteria")
    if qtype == "noul":
        if isinstance(criteria, dict):
            yes = criteria.get("true") or criteria.get("yes")
            no = criteria.get("false") or criteria.get("no")
            hints = []
            if yes:
                hints.append(f"Answer yes when: {yes}")
            if no:
                hints.append(f"Answer no when: {no}")
            if hints:
                instructions = f"{instructions}\n\n" + "\n".join(hints)
        return {"type": "predicate", "name": name, "instructions": instructions}
    if qtype == "choice":
        criteria = criteria or {}
        if isinstance(criteria, list):
            criteria = {str(entry): "" for entry in criteria}
        return {
            "type": "choice",
            "name": name,
            "instructions": instructions,
            "choices": [
                {"value": str(value), "description": str(desc or "")}
                for value, desc in criteria.items()
            ],
        }
    levels = []
    for level in criteria or []:
        text = str(level)
        label, sep, description = text.partition(":")
        levels.append(
            {
                "label": label.strip() or text,
                "description": (description.strip() or label.strip() or text) if sep else (label.strip() or text),
            }
        )
    return {"type": "score", "name": name, "instructions": instructions, "levels": levels}


def _gemini_question(name: str, question: Dict[str, Any]) -> Dict[str, Any]:
    qtype = question["type"]
    payload: Dict[str, Any] = {
        "type": qtype,
        "instructions": question["instructions"],
    }
    criteria = question.get("criteria")
    if criteria is not None:
        if qtype == "noul" and isinstance(criteria, dict):
            criteria = {
                "true_description": criteria.get("true", ""),
                "false_description": criteria.get("false", ""),
            }
        payload["criteria"] = criteria
    return payload


def _perplexity_question(name: str, question: Dict[str, Any]) -> Dict[str, Any]:
    # Perplexity's documented shape matches our canonical shape exactly.
    payload: Dict[str, Any] = {
        "type": question["type"],
        "instructions": question["instructions"],
    }
    if question.get("criteria") is not None:
        payload["criteria"] = question["criteria"]
    return payload


def build_payload(
    provider: str,
    state: Union[str, Dict[str, Any], List[Any]],
    questions: Union[Dict[str, Any], List[Any]],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the provider-specific JSON body for a decisions request."""
    provider = resolve_provider(provider)
    normalized = _normalize_questions(questions)
    effective_model = model or os.environ.get(_MODEL_ENV[provider]) or DEFAULT_MODELS[provider]

    if provider == "openai":
        body = {
            "model": effective_model,
            "input": _openai_input(state),
            "questions": [
                _openai_question(name, question) for name, question in normalized.items()
            ],
        }
    elif provider == "gemini":
        body = {
            "model": effective_model,
            "input": _gemini_input(state),
            "questions": {
                name: _gemini_question(name, question)
                for name, question in normalized.items()
            },
        }
    else:
        body = {
            "model": effective_model,
            "state": state if isinstance(state, (str, dict, list)) else _state_to_text(state),
            "questions": {
                name: _perplexity_question(name, question)
                for name, question in normalized.items()
            },
        }
    return body


# --------------------------------------------------------------------------- #
# response parsing
# --------------------------------------------------------------------------- #

def _probabilities_dict(raw: Any) -> Dict[str, float]:
    """Normalize a probabilities map or list-of-records into ``{key: prob}``."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    out: Dict[str, float] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        key = entry.get("value", entry.get("label"))
        if key is None:
            continue
        out[str(key)] = float(entry.get("probability", 0.0))
    return out


def _legend_from(raw: Any) -> Optional[Dict[str, str]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    legend: Dict[str, str] = {}
    for entry in raw:
        if isinstance(entry, dict) and entry.get("value") is not None:
            title = entry.get("label")
            legend[str(entry["value"])] = str(title if title is not None else entry["value"])
    return legend or None


def _parse_choice(answer: Dict[str, Any]) -> ChoiceResult:
    probabilities = _probabilities_dict(answer.get("probabilities"))
    choice = answer.get("choice")
    if choice is None and probabilities:
        choice = max(probabilities, key=probabilities.get)
    confidence = answer.get("confidence")
    if confidence is None:
        confidence = probabilities.get(str(choice), 0.0)
    return ChoiceResult(
        choice=str(choice if choice is not None else ""),
        probabilities=probabilities,
        confidence=float(confidence or 0.0),
    )


def _parse_noul(answer: Dict[str, Any]) -> NoulResult:
    value = answer.get("noul")
    if value is None:
        value = answer.get("probability")
    if value is None:
        probabilities = _probabilities_dict(answer.get("probabilities"))
        value = probabilities.get("yes", 0.5)
    return NoulResult(noul=float(max(0.0, min(1.0, float(value)))))


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _parse_score(answer: Dict[str, Any]) -> ScoreResult:
    probabilities = _probabilities_dict(answer.get("probabilities"))
    legend = _legend_from(answer.get("legend"))
    if legend is None and isinstance(answer.get("probabilities"), list):
        legend = _legend_from(answer.get("probabilities"))
    score_value = answer.get("score")
    if score_value is None and probabilities:
        total = sum(probabilities.values())
        if total > 0:
            score_value = (
                sum(float(k) * v for k, v in probabilities.items() if _is_number(k)) / total
            )
    confidence = answer.get("confidence")
    if confidence is None:
        confidence = max(probabilities.values(), default=0.0)
    return ScoreResult(
        score=float(score_value or 0.0),
        probabilities=probabilities,
        confidence=float(confidence or 0.0),
        legend=legend,
    )


_PARSERS = {"noul": _parse_noul, "choice": _parse_choice, "score": _parse_score}


def parse_answers(payload: Dict[str, Any], provider: Optional[str] = None) -> System1Result:
    """Convert a raw Decisions API payload into a :class:`System1Result`.

    Handles both response shapes: a map keyed by question name (Gemini,
    Perplexity) and an ordered list of answers carrying ``name`` (OpenAI).
    """
    raw = payload.get("answers")
    if raw is None:
        raise DecisionsError(f"Decisions response is missing 'answers': {payload!r}")

    entries: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, dict):
        entries = {str(name): answer for name, answer in raw.items()}
    elif isinstance(raw, list):
        for index, answer in enumerate(raw):
            if not isinstance(answer, dict):
                raise DecisionsError(f"Answer at index {index} is not an object: {answer!r}")
            name = answer.get("name") or f"question_{index}"
            entries[str(name)] = answer
    else:
        raise DecisionsError(f"Unexpected 'answers' payload: {raw!r}")

    answers: Dict[str, Any] = {}
    for name, answer in entries.items():
        if not isinstance(answer, dict):
            raise DecisionsError(f"Answer for {name!r} is not an object: {answer!r}")
        if answer.get("type") == "refusal":
            reason = answer.get("refusal") or answer.get("reason") or "no reason given"
            raise DecisionsError(f"Decision model refused question {name!r}: {reason}")
        qtype = _normalize_question_type(answer.get("type", "choice"))
        answers[name] = _PARSERS[qtype](answer)
    return System1Result(
        answers=answers,
        routing={"provider": provider, "model": payload.get("model"), "usage": payload.get("usage")},
    )


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #

@dataclass
class DecisionsConfig:
    """Configuration for a hosted Decisions API backend."""

    provider: str = "openai"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[float] = DEFAULT_TIMEOUT
    headers: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.provider = resolve_provider(self.provider)

    def resolved_api_key(self) -> Optional[str]:
        return self.api_key or os.environ.get(_API_KEY_ENV[self.provider])

    def resolved_model(self) -> str:
        return self.model or os.environ.get(_MODEL_ENV[self.provider]) or DEFAULT_MODELS[self.provider]

    def resolved_base_url(self) -> str:
        return (
            self.base_url
            or os.environ.get(_BASE_URL_ENV[self.provider])
            or DEFAULT_BASE_URLS[self.provider]
        ).rstrip("/")

    def endpoint(self) -> str:
        return f"{self.resolved_base_url()}/decisions"

    def auth_headers(self) -> Dict[str, str]:
        key = self.resolved_api_key()
        if not key:
            raise DecisionsError(
                f"No API key for provider {self.provider!r}. Set {_API_KEY_ENV[self.provider]} "
                f"or pass api_key=... ."
            )
        headers = {"Content-Type": "application/json"}
        if self.provider == "gemini":
            headers["x-goog-api-key"] = key
        else:
            headers["Authorization"] = f"Bearer {key}"
        headers.update(self.headers)
        return headers


def _extract_error(response: "requests.Response") -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
    text = (response.text or "").strip()
    return text[:500] if text else f"HTTP {response.status_code}"


def _raise_for_status(response: "requests.Response", provider: str, model: str) -> None:
    if response.ok:
        return
    detail = _extract_error(response)
    if response.status_code in (401, 403):
        raise DecisionsError(
            f"{provider} rejected the API key for decisions ({response.status_code}: {detail}). "
            f"Check {_API_KEY_ENV[provider]}."
        )
    if response.status_code == 404:
        raise DecisionsError(
            f"{provider} has no decisions endpoint at {response.url} (404: {detail}). "
            f"The Decisions API may not be enabled for model {model!r}."
        )
    if response.status_code == 429:
        raise DecisionsError(f"{provider} rate-limited the decisions request (429: {detail}).")
    if response.status_code == 400:
        raise DecisionsError(
            f"{provider} rejected the decisions request (400: {detail}). Confirm model "
            f"{model!r} is a decision model and the question shape is valid for this provider."
        )
    raise DecisionsError(f"{provider} decisions request failed ({response.status_code}): {detail}")


def decisions(
    state: Union[str, Dict[str, Any], List[Any]],
    questions: Union[Dict[str, Any], List[Any]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    headers: Optional[Dict[str, str]] = None,
    session: Optional["requests.Session"] = None,
) -> Dict[str, Any]:
    """POST one ``/decisions`` request and return the raw provider payload."""
    config = DecisionsConfig(
        provider=resolve_provider(provider),
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        headers=dict(headers or {}),
    )
    body = json.dumps(
        build_payload(config.provider, state, questions, model=config.resolved_model()),
        ensure_ascii=False,
    )
    requester = session or requests
    try:
        response = requester.post(
            config.endpoint(),
            data=body.encode("utf-8"),
            headers=config.auth_headers(),
            timeout=config.timeout,
        )
    except requests.exceptions.RequestException as exc:
        raise DecisionsError(
            f"Could not reach the {config.provider} Decisions API at {config.endpoint()} ({exc})."
        ) from exc
    _raise_for_status(response, config.provider, config.resolved_model())
    try:
        data = response.json()
    except ValueError as exc:
        raise DecisionsError(f"{config.provider} returned a non-JSON response: {exc}") from exc
    if not isinstance(data, dict):
        raise DecisionsError(f"Unexpected {config.provider} decisions response: {data!r}")
    return data


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #

class DecisionsClient:
    """Decisions predictor backed by a hosted Decisions API.

    Exposes the same ``choice``/``noul``/``score``/``predict`` surface as
    :class:`npcpy.gen.systemone.OllamaSystem1Client` and
    :class:`npcpy.ft.system1.System1Predictor`, so it can be dropped into
    :class:`npcpy.gen.decision.DecisionSystem1`.
    """

    def __init__(self, config: Optional[DecisionsConfig] = None, **kwargs):
        if config is None:
            config = DecisionsConfig(**kwargs)
        self.config = config
        self.session: Optional["requests.Session"] = None
        self.label_maps: Dict[str, List[str]] = {}

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.resolved_model()

    def _resolve_question_name(self, question_name: str) -> str:
        return question_name

    def _call(self, state: Any, questions: Any) -> System1Result:
        payload = decisions(
            state,
            questions,
            provider=self.config.provider,
            model=self.config.resolved_model(),
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers=self.config.headers,
            session=self.session,
        )
        return parse_answers(payload, provider=self.config.provider)

    def choice(
        self,
        state: Union[str, Dict[str, Any]],
        instructions: str,
        criteria: Dict[str, str],
        question_name: str = "default",
    ) -> ChoiceResult:
        result = self._call(
            state,
            {question_name: {"type": "choice", "instructions": instructions, "criteria": criteria}},
        )
        return result.answers[question_name]

    def noul(
        self,
        state: Union[str, Dict[str, Any]],
        instructions: str,
        criteria: Optional[Dict[str, str]] = None,
        question_name: str = "default",
    ) -> NoulResult:
        question: Dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria is not None:
            question["criteria"] = criteria
        result = self._call(state, {question_name: question})
        return result.answers[question_name]

    def score(
        self,
        state: Union[str, Dict[str, Any]],
        instructions: str,
        criteria: List[str],
        question_name: str = "default",
    ) -> ScoreResult:
        result = self._call(
            state,
            {question_name: {"type": "score", "instructions": instructions, "criteria": criteria}},
        )
        return result.answers[question_name]

    def predict(
        self,
        state: Union[str, Dict[str, Any]],
        questions: Union[Dict[str, Any], List[Any]],
    ) -> System1Result:
        return self._call(state, questions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.provider,
            "model": self.model,
            "base_url": self.config.resolved_base_url(),
            "api_key_env": _API_KEY_ENV[self.provider],
        }


def load_decisions(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> DecisionsClient:
    """Create a :class:`DecisionsClient` for a hosted provider."""
    config = DecisionsConfig(
        provider=resolve_provider(provider),
        model=model,
        api_key=api_key,
        base_url=base_url,
        **{k: v for k, v in kwargs.items() if k in {"timeout", "headers"}},
    )
    return DecisionsClient(config)


def available(provider: Optional[str] = None, timeout: float = 10) -> Dict[str, Any]:
    """Report whether a provider is configured.

    This is a local check only -- it confirms the provider is known, the API key
    is present, and a model name resolves. It does not spend tokens. Pass
    ``verify=True`` to :func:`require_available` for a live round-trip.
    """
    try:
        resolved = resolve_provider(provider)
    except DecisionsError as exc:
        return {
            "available": False,
            "provider": provider,
            "api_key_present": False,
            "model": None,
            "error": str(exc),
        }
    config = DecisionsConfig(provider=resolved, timeout=timeout)
    key = config.resolved_api_key()
    return {
        "available": bool(key),
        "provider": resolved,
        "api_key_present": bool(key),
        "api_key_env": _API_KEY_ENV[resolved],
        "model": config.resolved_model(),
        "base_url": config.resolved_base_url(),
        "error": None if key else f"{_API_KEY_ENV[resolved]} is not set",
    }


def require_available(
    provider: Optional[str] = None,
    verify: bool = False,
    timeout: float = 10,
) -> Dict[str, Any]:
    """Like :func:`available` but raises with a remediation hint when unusable."""
    info = available(provider=provider, timeout=timeout)
    if not info["available"]:
        if info.get("error", "").startswith("Unknown Decisions"):
            raise DecisionsError(info["error"])
        raise DecisionsError(
            f"No API key for provider {info['provider']!r}. Set {info['api_key_env']} "
            f"or pass api_key=... ."
        )
    if verify:
        client = load_decisions(provider=info["provider"], timeout=timeout)
        client.noul("ping", instructions="Is this a non-empty string?", question_name="probe")
        info["verified"] = True
    return info


# --------------------------------------------------------------------------- #
# module-level convenience wrappers
# --------------------------------------------------------------------------- #

def choice(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    criteria: Dict[str, str],
    provider: Optional[str] = None,
    client: Optional[DecisionsClient] = None,
    question_name: str = "default",
    **kwargs,
) -> ChoiceResult:
    client = client or load_decisions(provider=provider, **kwargs)
    return client.choice(state, instructions, criteria, question_name=question_name)


def noul(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    provider: Optional[str] = None,
    client: Optional[DecisionsClient] = None,
    criteria: Optional[Dict[str, str]] = None,
    question_name: str = "default",
    **kwargs,
) -> NoulResult:
    client = client or load_decisions(provider=provider, **kwargs)
    return client.noul(state, instructions, criteria=criteria, question_name=question_name)


def score(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    criteria: List[str],
    provider: Optional[str] = None,
    client: Optional[DecisionsClient] = None,
    question_name: str = "default",
    **kwargs,
) -> ScoreResult:
    client = client or load_decisions(provider=provider, **kwargs)
    return client.score(state, instructions, criteria, question_name=question_name)


def predict(
    state: Union[str, Dict[str, Any]],
    questions: Union[Dict[str, Any], List[Any]],
    provider: Optional[str] = None,
    client: Optional[DecisionsClient] = None,
    **kwargs,
) -> System1Result:
    client = client or load_decisions(provider=provider, **kwargs)
    return client.predict(state, questions)
