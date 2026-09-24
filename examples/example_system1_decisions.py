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
from npcpy.gen.decision import (
    DecisionQuestion,
    DecisionRouter,
    DecisionSystem1,
    decision_route,
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


def gen_decision_routing_example():
    examples = make_training_data()
    model = DecisionSystem1(
        config=System1Config(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            classifier="LogisticRegression",
            output_dir="./system1_decision_model",
        )
    )
    model.train(examples)
    print("decision model trained at", model.model_path)

    ticket = {
        "ticket": "Our production API has been returning HTTP 500 errors since this morning. Around 40% of customers cannot complete checkout.",
    }
    questions = [
        DecisionQuestion(
            name="queue",
            type="choice",
            instructions="What queue should handle this ticket?",
            criteria={
                "billing": "refunds, charges, invoices",
                "technical": "bugs, errors, outages",
                "sales": "purchasing, upgrades",
            },
        ),
        DecisionQuestion(
            name="urgency",
            type="score",
            instructions="How urgent is this ticket?",
            criteria=["low", "medium", "high", "critical"],
        ),
        DecisionQuestion(
            name="churn",
            type="noul",
            instructions="Does the customer express churn intent?",
        ),
    ]
    result = model.decide(ticket, questions)
    print("batch decision result:", result.to_dict())

    router = DecisionRouter(
        tiers=["self_service", "agent", "engineering"],
        criteria={
            "self_service": "user can resolve with a link or docs",
            "agent": "needs human support but no code change",
            "engineering": "requires a code or infrastructure fix",
        },
    )
    route = decision_route(
        ticket,
        instructions="Which team should handle this ticket?",
        tiers=router.tiers,
        criteria=router.criteria,
        model_path=model.model_path,
    )
    print("routing result:", route)


if __name__ == "__main__":
    train_and_predict()
    if os.path.isdir("./system1_ticket_model"):
        use_module_level_functions()
    gen_decision_routing_example()
