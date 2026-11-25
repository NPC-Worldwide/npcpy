"""Decision-module implementation for npcpy.

Mirrors ``npcpy.gen.response``: it exposes low-level provider-specific helpers
(``get_litellm_decision`` for hosted decision models and ``get_ollama_decision``
for local Ollama System One models) plus the high-level ``decision_request``
dispatcher. The public entrypoint is ``npcpy.llm_funcs.get_decision_response``,
which calls ``decision_request``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

try:
    import requests
except ImportError:
    requests = None  # type: ignore

try:
    import litellm
    from litellm import completion as _litellm_completion
    HAS_LITELLM = True
except ImportError:
    litellm = None  # type: ignore
    _litellm_completion = None  # type: ignore
    HAS_LITELLM = False

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    ollama = None  # type: ignore
    HAS_OLLAMA = False

logger = logging.getLogger(__name__)

# Providers that are expected to speak the System One / Decisions endpoint.
# Any provider not in this set is routed through the generic LiteLLM completion
# path with a System One-style body, or to Ollama for the ollama provider.
_DECISION_PROVIDERS = {
    "openai",
    "azure",
    "gemini",
    "perplexity",
    "typesafe",
    "openrouter",
    "fireworks_ai",
    "fireworks",
    "cloudflare",
    "databricks",
    "azure_ai",
    "litellm_proxy",
    "strands_decider",
    "hosted_vllm",
    "vllm",
    "laya",
    "bespoke",
}


class DecisionError(RuntimeError):
    """Raised when a decision request fails or is misconfigured."""


def _state_to_text(state: Union[str, Dict[str, Any], List[Any]]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, default=str)


def _normalize_question(
    question_type: str,
    instructions: str,
    criteria: Optional[Any] = None,
) -> Dict[str, Any]:
    qtype = str(question_type).strip().lower()
    if qtype not in {"choice", "score", "noul", "predicate"}:
        raise DecisionError(
            f"Unknown question type: {question_type!r}. "
            "Expected one of: choice, score, noul, predicate."
        )
    if qtype == "predicate":
        qtype = "noul"

    payload: Dict[str, Any] = {"type": qtype, "instructions": instructions or ""}
    if criteria is None:
        if qtype == "choice":
            raise DecisionError("choice question requires criteria mapping label -> description")
        if qtype == "score":
            raise DecisionError("score question requires criteria list of level descriptions")
        return payload

    if qtype == "choice":
        if not isinstance(criteria, dict):
            raise DecisionError("choice criteria must be a mapping of label -> description")
        payload["criteria"] = {str(k): "" if v is None else str(v) for k, v in criteria.items()}
    elif qtype == "score":
        if not isinstance(criteria, list):
            raise DecisionError("score criteria must be an ordered list of level descriptions")
        payload["criteria"] = [str(level) for level in criteria]
    else:
        if isinstance(criteria, dict):
            payload["criteria"] = criteria
    return payload


def _build_systemone_payload(
    model: str,
    state: Union[str, Dict[str, Any], List[Any]],
    question_type: str,
    instructions: str,
    criteria: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the System One request body (state + questions map)."""
    question = _normalize_question(question_type, instructions, criteria)
    return {
        "model": model,
        "state": state if isinstance(state, (str, dict, list)) else _state_to_text(state),
        "questions": {"default": question},
    }


def _build_decisions_payload(
    model: str,
    state: Union[str, Dict[str, Any], List[Any]],
    question_type: str,
    instructions: str,
    criteria: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the OpenAI Decisions request body (input + questions list)."""
    qtype = str(question_type).strip().lower()
    if qtype == "predicate":
        qtype = "noul"
    if qtype not in {"choice", "score", "noul"}:
        raise DecisionError(f"Unsupported decisions question type: {question_type!r}")

    question: Dict[str, Any] = {"type": qtype, "name": "default", "instructions": instructions or ""}
    if criteria is not None:
        if qtype == "choice":
            if not isinstance(criteria, dict):
                raise DecisionError("choice criteria must be a mapping")
            question["choices"] = [
                {"value": k, "description": "" if v is None else str(v)}
                for k, v in criteria.items()
            ]
        elif qtype == "score":
            if not isinstance(criteria, list):
                raise DecisionError("score criteria must be a list")
            question["levels"] = [
                {"label": str(level).split(":", 1)[0].strip(),
                 "description": (str(level).split(":", 1)[1].strip() if ":" in str(level) else str(level))}
                for level in criteria
            ]
        else:
            if isinstance(criteria, dict):
                question["criteria"] = criteria

    input_value = state if isinstance(state, (str, list)) else _state_to_text(state)
    return {
        "model": model,
        "input": input_value,
        "questions": [question],
    }


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _probabilities(raw: Any) -> Dict[str, float]:
    """Normalize probabilities from various provider shapes."""
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items() if v is not None}
    if isinstance(raw, list):
        out: Dict[str, float] = {}
        for entry in raw:
            if isinstance(entry, dict):
                value = entry.get("value")
                label = entry.get("label")
                key = label if label is not None else value
                if key is None:
                    continue
                prob = entry.get("probability") or entry.get("prob")
                if prob is None:
                    continue
                out[str(key)] = float(prob)
        return out
    return {}


def _parse_answer(answer: Dict[str, Any]) -> Dict[str, Any]:
    qtype = str(answer.get("type", "choice")).strip().lower()
    if qtype == "predicate":
        qtype = "noul"

    probs = _probabilities(answer.get("probabilities"))
    confidence = answer.get("confidence")
    if confidence is None and probs:
        confidence = max(probs.values(), default=0.0)

    if qtype == "choice":
        return {
            "type": "choice",
            "choice": answer.get("choice"),
            "confidence": float(confidence or 0.0),
            "probabilities": probs,
        }
    if qtype == "noul":
        noul = answer.get("noul")
        if noul is None and probs:
            noul = probs.get("true", 0.0)
        return {
            "type": "noul",
            "noul": float(noul or 0.0),
            "confidence": float(confidence or 0.0),
            "probabilities": probs,
        }
    if qtype == "score":
        score = answer.get("score")
        if score is None and probs:
            total = sum(probs.values())
            if total > 0:
                score = sum(float(k) * v for k, v in probs.items() if _is_number(k)) / total
        return {
            "type": "score",
            "score": float(score or 0.0),
            "confidence": float(confidence or 0.0),
            "probabilities": probs,
        }
    return {"type": qtype, "confidence": float(confidence or 0.0), "probabilities": probs}


def _parse_systemone_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = payload.get("answers")
    if not isinstance(raw, dict):
        raise DecisionError(f"System One response missing answers map: {payload!r}")
    answer = raw.get("default")
    if not isinstance(answer, dict):
        raise DecisionError(f"Missing answer for 'default' question: {raw!r}")
    if answer.get("type") == "refusal":
        raise DecisionError(f"Decision model refused: {answer.get('refusal', 'no reason')}")
    parsed = _parse_answer(answer)
    parsed["model"] = payload.get("model")
    parsed["usage"] = payload.get("usage")
    return parsed


def _parse_decisions_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = payload.get("answers")
    if isinstance(raw, list):
        answer = next((a for a in raw if a.get("name") == "default"), raw[0] if raw else None)
    elif isinstance(raw, dict):
        answer = raw.get("default")
    else:
        answer = None
    if not isinstance(answer, dict):
        raise DecisionError(f"Decisions response missing answers: {payload!r}")
    if answer.get("type") == "refusal":
        raise DecisionError(f"Decision model refused: {answer.get('refusal', 'no reason')}")
    parsed = _parse_answer(answer)
    parsed["model"] = payload.get("model")
    parsed["usage"] = payload.get("usage")
    return parsed


# --------------------------------------------------------------------------- #
# Provider-specific helpers
# --------------------------------------------------------------------------- #

# Provider defaults for direct HTTP fallback when litellm.systemone is unavailable.
_HOSTED_DECISION_DEFAULTS = {
    "openai": {"api_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
    "gemini": {"api_url": "https://generativelanguage.googleapis.com/v1beta", "api_key_env": "GEMINI_API_KEY"},
    "perplexity": {"api_url": "https://api.perplexity.ai/v1", "api_key_env": "PERPLEXITY_API_KEY"},
}


def _resolve_api_key(provider: str, api_key: Optional[str]) -> Optional[str]:
    if api_key:
        return api_key
    info = _HOSTED_DECISION_DEFAULTS.get(provider.lower())
    if info:
        return os.environ.get(info["api_key_env"])
    return None


def _resolve_api_url(provider: str, api_url: Optional[str]) -> str:
    if api_url:
        return api_url.rstrip("/")
    info = _HOSTED_DECISION_DEFAULTS.get(provider.lower())
    if info:
        return info["api_url"].rstrip("/")
    raise DecisionError(f"No default API URL for provider {provider!r}; pass api_url=")


def _direct_http_decision(
    provider: str,
    model: str,
    payload: Dict[str, Any],
    api_key: Optional[str],
    api_url: Optional[str],
    timeout: float = 120,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """POST a System One decision request directly to the provider endpoint."""
    url = f"{_resolve_api_url(provider, api_url)}/systemone"
    key = _resolve_api_key(provider, api_key)
    if not key:
        info = _HOSTED_DECISION_DEFAULTS.get(provider.lower(), {})
        raise DecisionError(
            f"No API key for provider {provider!r}. "
            f"Set {info.get('api_key_env', 'the provider API key env var')} or pass api_key=."
        )

    request_headers = {"Content-Type": "application/json"}
    if provider.lower() == "gemini":
        request_headers["x-goog-api-key"] = key
    else:
        request_headers["Authorization"] = f"Bearer {key}"
    if headers:
        request_headers.update(headers)

    try:
        response = requests.post(
            url,
            json=payload,
            headers=request_headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        raise DecisionError(f"Direct HTTP decision call failed for {provider}: {exc}") from exc


def get_litellm_decision(
    prompt: Union[str, Dict[str, Any], List[Any]],
    model: str,
    provider: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    question_type: str = "choice",
    instructions: Optional[str] = None,
    criteria: Optional[Any] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Call a hosted decision model through LiteLLM's /v1/systemone route.

    ``provider`` should be a LiteLLM-supported decision provider such as
    ``openai``, ``gemini``, ``perplexity``, ``typesafe``, etc. ``model`` is the
    decision model name. The System One request body is used because it is the
    closest to the npcpy.ft.system1 surface.
    """
    if not HAS_LITELLM:
        raise DecisionError("litellm is required for hosted decision providers")

    instructions = instructions or "Choose the best answer."
    qualified = model if "/" in model else f"{provider}/{model}"

    payload = _build_systemone_payload(
        model=qualified,
        state=prompt,
        question_type=question_type,
        instructions=instructions,
        criteria=criteria,
    )

    api_params: Dict[str, Any] = {
        "model": qualified,
        "state": payload["state"],
        "questions": payload["questions"],
    }
    if api_key is not None:
        api_params["api_key"] = api_key
    if api_url is not None:
        api_params["api_base"] = api_url
    for key, value in kwargs.items():
        if key in {"timeout", "headers", "extra_headers"}:
            api_params[key] = value

    try:
        if hasattr(litellm, "systemone"):
            resp = litellm.systemone(**api_params)
        elif hasattr(litellm, "main") and hasattr(litellm.main, "systemone"):
            resp = litellm.main.systemone(**api_params)
        else:
            # Direct HTTP fallback to the provider's /v1/systemone endpoint.
            try:
                raw = _direct_http_decision(
                    provider=provider,
                    model=model,
                    payload=payload,
                    api_key=api_key,
                    api_url=api_url,
                    timeout=kwargs.get("timeout", 120),
                    headers=kwargs.get("headers") or kwargs.get("extra_headers"),
                )
                return _parse_systemone_response(raw)
            except Exception as http_exc:
                # Last resort: generic completion (loses probabilities).
                messages = [
                    {"role": "user", "content": f"{instructions}\n\n{_state_to_text(prompt)}"}
                ]
                if criteria:
                    messages[0]["content"] += f"\n\nOptions: {json.dumps(criteria)}"
                completion_params = {
                    "model": qualified,
                    "messages": messages,
                    "temperature": 0.0,
                }
                if api_key is not None:
                    completion_params["api_key"] = api_key
                if api_url is not None:
                    completion_params["api_base"] = api_url
                for key, value in kwargs.items():
                    if key in {"timeout", "headers", "extra_headers", "temperature"}:
                        completion_params[key] = value
                try:
                    resp = _litellm_completion(**completion_params)
                    text = resp.choices[0].message.content if resp.choices else ""
                    return {
                        "type": "choice",
                        "choice": text.strip() if text else None,
                        "confidence": 1.0 if text else 0.0,
                        "probabilities": {},
                        "model": qualified,
                        "raw": str(text),
                    }
                except Exception as comp_exc:
                    raise DecisionError(
                        f"Decision HTTP fallback failed ({http_exc}); generic completion also failed ({comp_exc})"
                    ) from comp_exc
    except Exception as e:
        raise DecisionError(f"litellm decision call failed: {e}") from e

    raw = getattr(resp, "model_dump", lambda: resp.__dict__)() if hasattr(resp, "model_dump") or hasattr(resp, "__dict__") else resp
    return _parse_systemone_response(raw)


def get_ollama_decision(
    prompt: Union[str, Dict[str, Any], List[Any]],
    model: str,
    api_url: Optional[str] = None,
    question_type: str = "choice",
    instructions: Optional[str] = None,
    criteria: Optional[Any] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Call a local Ollama System One decision model.

    Sends a System One payload to ``{api_url}/v1/systemone``. Requires an
    Ollama version that supports the System One endpoint (>= 0.35.0) and a
    decision model such as tev1 or bespoke/nimble.
    """
    if not HAS_OLLAMA and requests is None:
        raise DecisionError("ollama or requests is required for ollama decision provider")

    base_url = (api_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    instructions = instructions or "Choose the best answer."

    payload = _build_systemone_payload(
        model=model,
        state=prompt,
        question_type=question_type,
        instructions=instructions,
        criteria=criteria,
    )

    if HAS_OLLAMA and hasattr(ollama, "systemone"):
        try:
            resp = ollama.systemone(model=model, state=payload["state"], questions=payload["questions"])
            return _parse_systemone_response(resp)
        except Exception:
            pass

    if requests is None:
        raise DecisionError("requests is required to call Ollama System One endpoint")

    url = f"{base_url}/v1/systemone"
    headers = {"Content-Type": "application/json"}
    extra_headers = kwargs.get("headers") or kwargs.get("extra_headers")
    if isinstance(extra_headers, dict):
        headers.update(extra_headers)
    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=kwargs.get("timeout", 120),
        )
        response.raise_for_status()
        return _parse_systemone_response(response.json())
    except Exception as e:
        raise DecisionError(f"Ollama decision call failed: {e}") from e


# --------------------------------------------------------------------------- #
# High-level dispatcher (internal, called from npcpy.llm_funcs)
# --------------------------------------------------------------------------- #

def decision_request(
    prompt: Union[str, Dict[str, Any], List[Any]],
    model: str,
    provider: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    question_type: str = "choice",
    instructions: Optional[str] = None,
    criteria: Optional[Any] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Dispatch a decision request to the appropriate provider-specific helper."""
    if not model:
        raise DecisionError("model is required for decision_request")
    if not provider:
        raise DecisionError("provider is required for decision_request")

    key = str(provider).strip().lower()
    if key in {"ollama", "systemone", "system_one", "system-1", "jev"}:
        return get_ollama_decision(
            prompt, model, api_url=api_url,
            question_type=question_type, instructions=instructions, criteria=criteria,
            api_key=api_key, **kwargs,
        )
    return get_litellm_decision(
        prompt, model, provider,
        api_key=api_key, api_url=api_url,
        question_type=question_type, instructions=instructions, criteria=criteria,
        **kwargs,
    )


def train_decision_model(
    examples: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    **kwargs,
) -> str:
    """Train a local system1 decision model from example records."""
    from npcpy.ft.system1 import System1Config, System1Example, train_system1

    records = []
    for ex in examples:
        state = ex.get("state")
        if state is None:
            continue
        answer = ex.get("answer")
        question_type = ex.get("type", "choice")
        if isinstance(answer, str):
            if question_type == "choice":
                answer = {"choice": answer}
            elif question_type == "score":
                answer = {"score": int(answer)}
            elif question_type == "noul":
                answer = {"noul": float(answer)}
        records.append(
            System1Example(
                state=state,
                question_type=question_type,
                instructions=ex.get("instructions", ""),
                criteria=ex.get("criteria"),
                answer=answer,
                question_name=ex.get("name", "default"),
            )
        )

    config = System1Config(output_dir=output_dir or "./decision_model", **kwargs)
    return train_system1(records, config)


def load_decision_model(path: str) -> Any:
    from npcpy.ft.system1 import load_system1
    return load_system1(path)


def load_decision_examples(
    path: str,
    state_key: str = "state",
    name_key: str = "name",
    type_key: str = "type",
    instructions_key: str = "instructions",
    criteria_key: str = "criteria",
    answer_key: str = "answer",
) -> List[Dict[str, Any]]:
    """Load decision examples from a JSON/JSONL/CSV file."""
    import csv

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r") as f:
            records = json.load(f)
    elif ext == ".jsonl":
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif ext == ".csv":
        records = []
        with open(path, "r", newline="") as f:
            for row in csv.DictReader(f):
                parsed = {}
                for k, v in row.items():
                    if not v:
                        continue
                    try:
                        parsed[k] = json.loads(v)
                    except Exception:
                        parsed[k] = v
                records.append(parsed)
    else:
        raise DecisionError(f"Unsupported decision examples format: {ext}")

    if not isinstance(records, list):
        raise DecisionError("Decision examples file must contain a list of records")

    out = []
    for record in records:
        answer = record.get(answer_key)
        question_type = record.get(type_key, "choice")
        if isinstance(answer, str):
            if question_type == "choice":
                answer = {"choice": answer}
            elif question_type == "score":
                answer = {"score": int(answer)}
            elif question_type == "noul":
                answer = {"noul": float(answer)}
        out.append(
            {
                "state": record.get(state_key),
                "name": record.get(name_key, "default"),
                "type": question_type,
                "instructions": record.get(instructions_key, ""),
                "criteria": record.get(criteria_key),
                "answer": answer,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Flat convenience wrappers (no classes)
# --------------------------------------------------------------------------- #

def decision_choice(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    criteria: Dict[str, str],
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Single choice decision; returns a plain dict."""
    return decision_request(
        prompt=state,
        model=model or "",
        provider=provider or "",
        api_key=api_key,
        api_url=api_url,
        question_type="choice",
        instructions=instructions,
        criteria=criteria,
        **kwargs,
    )


def decision_score(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    criteria: List[str],
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Single score decision; returns a plain dict."""
    return decision_request(
        prompt=state,
        model=model or "",
        provider=provider or "",
        api_key=api_key,
        api_url=api_url,
        question_type="score",
        instructions=instructions,
        criteria=criteria,
        **kwargs,
    )


def decision_noul(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Single yes/no decision; returns a plain dict."""
    return decision_request(
        prompt=state,
        model=model or "",
        provider=provider or "",
        api_key=api_key,
        api_url=api_url,
        question_type="noul",
        instructions=instructions,
        **kwargs,
    )


def decision_predict(
    state: Union[str, Dict[str, Any]],
    questions: List[Dict[str, Any]],
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Predict multiple named questions at once; returns a plain dict of answers.

    Only a single named question is supported by the current flat implementation;
    the first question in the list is used.
    """
    if not questions:
        raise DecisionError("questions list is empty")
    q = questions[0]
    return decision_request(
        prompt=state,
        model=model or "",
        provider=provider or "",
        api_key=api_key,
        api_url=api_url,
        question_type=q.get("type", "choice"),
        instructions=q.get("instructions", ""),
        criteria=q.get("criteria"),
        **kwargs,
    )


def decision_route(
    state: Union[str, Dict[str, Any]],
    instructions: str,
    tiers: List[str],
    criteria: Optional[Dict[str, str]] = None,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Route state to one of several tiers using a choice decision."""
    if not tiers:
        raise DecisionError("tiers list is empty")
    crit = criteria or {tier: tier for tier in tiers}
    result = decision_choice(
        state=state,
        instructions=instructions,
        criteria=crit,
        model=model,
        provider=provider,
        api_url=api_url,
        api_key=api_key,
        **kwargs,
    )
    selected = result.get("choice") or tiers[-1]
    confidence = result.get("confidence", 0.0)
    if threshold is not None and confidence < threshold:
        selected = tiers[-1]
    return {
        "tier": selected,
        "confidence": confidence,
        "probabilities": result.get("probabilities", {}),
        "tiers": tiers,
    }
