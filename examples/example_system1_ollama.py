"""System One decisions via a local Ollama decision model (``/v1/systemone``).

Requires Ollama >= 0.35.0 and a pulled decision model::

    ollama pull nimble      # 9B, from Bespoke Labs
    # or: ollama pull tev1  /  ollama pull tev1:0.8b

No API key is needed; everything runs on this machine.
"""

import json
import os

from npcpy.gen.systemone import (
    DEFAULT_MODEL,
    OllamaSystem1Client,
    OllamaSystem1Config,
    available,
    systemone,
)
from npcpy.gen.decision import (
    DecisionQuestion,
    DecisionRouter,
    DecisionSystem1,
    decision_route,
)

TICKET = {
    "ticket": "I was charged twice. Please refund the extra payment.",
    "customer": {"plan": "pro", "tenure_months": 4},
}

QUEUE_CRITERIA = {
    "billing": "Payments and refunds",
    "technical": "Bugs and integrations",
    "other": "None of the above",
}

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": QUEUE_CRITERIA,
    },
    "refund": {
        "type": "noul",
        "instructions": "Does the customer explicitly ask for a refund?",
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["Routine", "Soon", "Urgent"],
    },
}


def resolve_model(model=None):
    """Honor NPCPY_SYSTEM1_OLLAMA_MODEL, the same env var the library reads."""
    return model or os.environ.get("NPCPY_SYSTEM1_OLLAMA_MODEL") or DEFAULT_MODEL


def check_environment(model=None):
    model = resolve_model(model)
    info = available(model=model)
    print("ollama check:", json.dumps(info, indent=2))
    if not info["available"]:
        reason = info["error"] or (
            "upgrade Ollama to >= 0.35.0" if not info["version_ok"] else f"run `ollama pull {model}`"
        )
        raise SystemExit(f"System One is not ready: {reason}")
    return info


def single_request_example(model=None):
    model = resolve_model(model)
    """All three question types are scored in one POST."""
    payload = systemone(TICKET, QUESTIONS, model=model)
    print("raw response:", json.dumps(payload, indent=2)[:2000])


def typed_client_example(model=None):
    model = resolve_model(model)
    client = OllamaSystem1Client(OllamaSystem1Config(model=model))
    result = client.predict(TICKET, QUESTIONS)

    team = result.answers["team"]
    refund = result.answers["refund"]
    urgency = result.answers["urgency"]

    print(f"team:    {team.choice} (confidence {team.confidence:.3f}) {team.probabilities}")
    print(f"refund:  {refund.noul:.3f} (P(true))")
    print(f"urgency: {urgency.score:.3f} over {urgency.legend}")
    return result


def decision_helper_example(model=None):
    model = resolve_model(model)
    """The gen.decision surface forwards to Ollama with backend='ollama'."""
    model_helper = DecisionSystem1.from_ollama(model=model)

    ticket_questions = [
        DecisionQuestion(name="team", type="choice",
                         instructions="Which team should handle this ticket?",
                         criteria=QUEUE_CRITERIA),
        DecisionQuestion(name="refund", type="noul",
                         instructions="Does the customer explicitly ask for a refund?"),
        DecisionQuestion(name="urgency", type="score",
                         instructions="How urgent is this ticket?",
                         criteria=["Routine", "Soon", "Urgent"]),
    ]
    result = model_helper.decide(TICKET, ticket_questions)
    print("decide():", json.dumps(result.to_dict(), indent=2))

    router = DecisionRouter(
        tiers=["self_service", "agent", "engineering"],
        criteria={
            "self_service": "the customer can resolve it with docs",
            "agent": "a human should respond, no code change",
            "engineering": "requires a code or infrastructure fix",
        },
    )
    route = decision_route(
        TICKET,
        instructions="Which team should handle this ticket?",
        tiers=router.tiers,
        criteria=router.criteria,
        threshold=0.6,
        backend="ollama",
        model=model,
    )
    print("route():", route)
    return result


if __name__ == "__main__":
    model = resolve_model()
    print(f"using model: {model}")
    try:
        check_environment(model)
    except SystemExit as exc:
        print(exc)
        print("(Showing the request shape only: skipping live calls.)")
        print(json.dumps({"model": model, "state": TICKET, "questions": QUESTIONS}, indent=2))
        raise SystemExit(0)

    single_request_example(model)
    typed_client_example(model)
    decision_helper_example(model)
