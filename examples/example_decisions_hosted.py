"""Hosted Decisions APIs: OpenAI, Gemini, and Perplexity.

All three providers answer the same ``noul`` / ``choice`` / ``score`` questions
and return the same result objects as the local Ollama System One backend
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
import sys

from npcpy.gen.decisions import (
    DEFAULT_MODELS,
    PROVIDERS,
    DecisionsClient,
    DecisionsConfig,
    DecisionsError,
    available,
)
from npcpy.gen.decision import DecisionQuestion, DecisionRouter, DecisionSystem1, decision_route

TICKET = {
    "ticket": "I was charged twice. Please refund the extra payment.",
    "customer": {"plan": "pro", "tenure_months": 4},
}

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


def check_environment(provider):
    info = available(provider=provider)
    print(f"{provider} check:", json.dumps(info, indent=2))
    if not info["available"]:
        raise SystemExit(f"{provider} is not configured: {info['error']}")
    return info


def typed_client_example(provider):
    client = DecisionsClient(DecisionsConfig(provider=provider))
    result = client.predict(TICKET, QUESTIONS)

    team = result.answers["team"]
    refund = result.answers["refund"]
    urgency = result.answers["urgency"]

    print(f"  team:    {team.choice} (confidence {team.confidence:.3f}) {team.probabilities}")
    print(f"  refund:  {refund.noul:.3f} (P(true))")
    print(f"  urgency: {urgency.score:.3f} over {urgency.legend}")
    return result


def unified_surface_example(provider):
    """The exact same decide()/route() calls work for every backend."""
    model = DecisionSystem1.from_hosted(provider=provider)
    result = model.decide(TICKET, QUESTIONS)
    print("  decide():", json.dumps(result.to_dict()["answers"], indent=2)[:600])

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
        backend=provider,
    )
    print("  route():", route)
    return result


def main(argv):
    requested = argv[1:] or list(PROVIDERS)
    unknown = [name for name in requested if name not in PROVIDERS]
    if unknown:
        raise SystemExit(f"unknown provider(s) {unknown}; expected one of {list(PROVIDERS)}")

    ran = 0
    for provider in requested:
        info = available(provider=provider)
        if not info["available"]:
            print(f"skipping {provider}: {info['error']}")
            continue
        print(f"\n=== {provider} ({DEFAULT_MODELS[provider]}) ===")
        check_environment(provider)
        try:
            typed_client_example(provider)
            unified_surface_example(provider)
        except DecisionsError as exc:
            print(f"  request failed: {exc}")
            continue
        ran += 1

    if not ran:
        raise SystemExit("no providers were configured; set an API key and retry")
    print(f"\nran {ran} provider(s)")


if __name__ == "__main__":
    main(sys.argv)
