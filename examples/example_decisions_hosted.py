"""Hosted Decisions APIs via the flat npcpy Decision Module.

All supported providers answer ``noul`` / ``choice`` / ``score`` questions and
return the same plain dict shape as the local Ollama backend
(see ``example_system1_ollama.py``), so switching backends is a one-line change.

API keys are read from the environment::

    export GEMINI_API_KEY=...
    export PERPLEXITY_API_KEY=...
    export OPENAI_API_KEY=...

Run with no arguments to exercise every provider whose key is set::

    python examples/example_decisions_hosted.py
    python examples/example_decisions_hosted.py gemini
"""

import json
import os
import sys

from npcpy.llm_funcs import get_decision_response
from npcpy.gen.decision import decision_predict, decision_route

# provider -> (model, env var)
PROVIDER_MODELS = {
    "openai": ("gpt-6-luna", "OPENAI_API_KEY"),
    "gemini": ("gemini-decision-1", "GEMINI_API_KEY"),
    "perplexity": ("pplx-decider-latest", "PERPLEXITY_API_KEY"),
}

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


def single_question_example(provider, model):
    """Use the public ``get_decision_response`` entrypoint for one question."""
    result = get_decision_response(
        prompt=TICKET,
        provider=provider,
        model=model,
        question_type="choice",
        instructions="Which team should handle this ticket?",
        criteria=QUEUE_CRITERIA,
    )
    print(f"  get_decision_response:", json.dumps(result["response"], indent=2)[:600])
    return result


def batch_example(provider, model):
    """Use the lower-level ``decision_predict`` helper for several questions."""
    result = decision_predict(
        TICKET,
        list(QUESTIONS.values()),
        provider=provider,
        model=model,
    )
    print(f"  decision_predict:", json.dumps(result, indent=2)[:600])
    return result


def route_example(provider, model):
    """Route the ticket to the best tier."""
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
        provider=provider,
        model=model,
    )
    print("  decision_route:", route)
    return route


def _provider_available(provider):
    """Check the conventional env var for the provider."""
    _, key_env = PROVIDER_MODELS[provider]
    return bool(os.environ.get(key_env))


def main(argv):
    requested = argv[1:] or list(PROVIDER_MODELS)
    unknown = [name for name in requested if name not in PROVIDER_MODELS]
    if unknown:
        raise SystemExit(
            f"unknown provider(s) {unknown}; expected one of {list(PROVIDER_MODELS)}"
        )

    ran = 0
    for provider in requested:
        model, key_env = PROVIDER_MODELS[provider]
        if not _provider_available(provider):
            print(f"skipping {provider}: {key_env} is not set")
            continue
        print(f"\n=== {provider} ({model}) ===")
        try:
            single_question_example(provider, model)
            batch_example(provider, model)
            route_example(provider, model)
        except Exception as exc:
            print(f"  request failed: {exc}")
            continue
        ran += 1

    if not ran:
        raise SystemExit("no providers were configured; set an API key and retry")
    print(f"\nran {ran} provider(s)")


if __name__ == "__main__":
    main(sys.argv)
