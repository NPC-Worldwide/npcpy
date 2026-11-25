"""System One decisions via a local Ollama decision model (``/v1/systemone``).

Requires Ollama >= 0.35.0 and a pulled decision model::

    ollama pull nimble      # 9B, from Bespoke Labs
    # or: ollama pull tev1  /  ollama pull tev1:0.8b

No API key is needed; everything runs on this machine.
"""

import json
import os
import sys

from npcpy.llm_funcs import get_decision_response
from npcpy.gen.decision import decision_predict, decision_route

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
    return model or os.environ.get("NPCPY_SYSTEM1_OLLAMA_MODEL") or "tev1:0.8b"


def single_question_example(model=None):
    """Single choice decision through the public entrypoint."""
    model = resolve_model(model)
    result = get_decision_response(
        prompt=TICKET,
        provider="ollama",
        model=model,
        question_type="choice",
        instructions="Which team should handle this ticket?",
        criteria=QUEUE_CRITERIA,
    )
    print("single question:", json.dumps(result["response"], indent=2))


def batch_example(model=None):
    """Ask several questions at once with the lower-level helper."""
    model = resolve_model(model)
    result = decision_predict(
        TICKET,
        list(QUESTIONS.values()),
        provider="ollama",
        model=model,
    )
    print("batch:", json.dumps(result, indent=2)[:1200])


def route_example(model=None):
    """Route the ticket to the best support tier."""
    model = resolve_model(model)
    route = decision_route(
        TICKET,
        instructions="Which team should handle this ticket?",
        tiers=["self_service", "agent", "engineering"],
        criteria={
            "self_service": "the customer can resolve it with docs",
            "agent": "a human should respond, no code change",
            "engineering": "requires a code or infrastructure fix",
        },
        threshold=0.6,
        provider="ollama",
        model=model,
    )
    print("route:", route)


if __name__ == "__main__":
    model = resolve_model()
    print(f"using model: {model}")
    single_question_example(model)
    batch_example(model)
    route_example(model)
