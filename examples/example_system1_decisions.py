"""Train a local npcpy.ft.system1 decision model and query it.

This example uses the local sentence-transformers + sklearn backend, not a
hosted decision API. It is useful when you want a small classifier that runs
offline once trained.
"""

import os

from npcpy.ft.system1 import (
    System1Config,
    System1Example,
    train_system1,
    load_system1,
    choice,
    noul,
    score,
    predict,
)


def make_training_data():
    return [
        System1Example(
            state={"ticket": "I was charged twice, please refund."},
            question_type="choice",
            instructions="What queue should handle this ticket?",
            criteria={
                "billing": "refunds, charges, invoices",
                "technical": "bugs, errors, outages",
                "sales": "purchasing, upgrades",
            },
            answer={"choice": "billing"},
            question_name="queue",
        ),
        System1Example(
            state={"ticket": "The API returns 500 on every request since this morning."},
            question_type="choice",
            instructions="What queue should handle this ticket?",
            criteria={
                "billing": "refunds, charges, invoices",
                "technical": "bugs, errors, outages",
                "sales": "purchasing, upgrades",
            },
            answer={"choice": "technical"},
            question_name="queue",
        ),
        System1Example(
            state={"ticket": "Production is completely unavailable and payments are failing."},
            question_type="score",
            instructions="How urgent is this ticket?",
            criteria=["low", "medium", "high", "critical"],
            answer={"score": 3},
            question_name="urgency",
        ),
        System1Example(
            state={"ticket": "Can you update my profile picture?"},
            question_type="score",
            instructions="How urgent is this ticket?",
            criteria=["low", "medium", "high", "critical"],
            answer={"score": 0},
            question_name="urgency",
        ),
        System1Example(
            state={"ticket": "I want to cancel my subscription immediately."},
            question_type="noul",
            instructions="Does the customer express churn intent?",
            answer={"noul": 1.0},
            question_name="churn",
        ),
        System1Example(
            state={"ticket": "Thanks for the great support this week."},
            question_type="noul",
            instructions="Does the customer express churn intent?",
            answer={"noul": 0.0},
            question_name="churn",
        ),
    ]


def train_and_predict():
    examples = make_training_data()
    config = System1Config(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        classifier="LogisticRegression",
        output_dir="./system1_ticket_model",
    )
    model_path = train_system1(examples, config)
    print(f"trained model at {model_path}")

    predictor = load_system1(model_path)

    ticket = {
        "ticket": "Our production API has been returning HTTP 500 errors since this morning. Around 40% of customers cannot complete checkout.",
    }

    queue_result = predictor.choice(
        ticket,
        instructions="What queue should handle this ticket?",
        criteria={
            "billing": "refunds, charges, invoices",
            "technical": "bugs, errors, outages",
            "sales": "purchasing, upgrades",
        },
        question_name="queue",
    )
    print("queue choice:", queue_result.choice, "confidence:", queue_result.confidence)

    urgency_result = predictor.score(
        ticket,
        instructions="How urgent is this ticket?",
        criteria=["low", "medium", "high", "critical"],
        question_name="urgency",
    )
    print("urgency score:", urgency_result.score, "probabilities:", urgency_result.probabilities)

    churn_result = predictor.noul(
        ticket,
        instructions="Does the customer express churn intent?",
        question_name="churn",
    )
    print("churn risk:", churn_result.noul)

    batch = predictor.predict(
        ticket,
        questions={
            "queue": {"type": "choice", "instructions": "What queue should handle this ticket?", "criteria": {
                "billing": "refunds, charges, invoices",
                "technical": "bugs, errors, outages",
                "sales": "purchasing, upgrades",
            }},
            "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["low", "medium", "high", "critical"]},
            "churn": {"type": "noul", "instructions": "Does the customer express churn intent?"},
        },
    )
    print("batch result:", batch.to_dict())


def use_module_level_functions():
    os.environ["NPCPY_SYSTEM1_MODEL"] = "./system1_ticket_model"

    ticket = {"ticket": "Please refund the duplicate charge on my account."}
    result = choice(
        ticket,
        instructions="What queue should handle this ticket?",
        criteria={
            "billing": "refunds, charges, invoices",
            "technical": "bugs, errors, outages",
            "sales": "purchasing, upgrades",
        },
    )
    print("module-level choice:", result.choice, "confidence:", result.confidence)

    risk = noul(ticket, instructions="Does the customer express churn intent?")
    print("module-level churn:", risk.noul)

    del os.environ["NPCPY_SYSTEM1_MODEL"]


if __name__ == "__main__":
    train_and_predict()
    if os.path.isdir("./system1_ticket_model"):
        use_module_level_functions()
