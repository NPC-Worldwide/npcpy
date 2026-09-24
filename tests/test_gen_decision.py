import json
import os
import shutil
import tempfile

import pytest

from npcpy.gen.decision import (
    DecisionQuestion,
    DecisionRouter,
    DecisionSystem1,
    decision_choice,
    decision_noul,
    decision_predict,
    decision_route,
    decision_score,
    load_decision_examples,
    load_decision_model,
    train_decision_model,
)
from npcpy.ft.system1 import System1Config, System1Example


@pytest.fixture
def ticket_examples():
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
            state={"ticket": "The API returns 500 on every request."},
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
            state={"ticket": "Production is completely unavailable."},
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


def test_train_and_decide(ticket_examples):
    tmpdir = tempfile.mkdtemp()
    try:
        config = System1Config(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            classifier="LogisticRegression",
            output_dir=tmpdir,
        )
        model = DecisionSystem1(config=config)
        path = model.train(ticket_examples)
        assert os.path.isdir(path)
        assert model.predictor is not None

        ticket = {"ticket": "The database is down and users cannot log in."}
        queue = model.choice(
            ticket,
            instructions="What queue should handle this ticket?",
            criteria={
                "billing": "refunds, charges, invoices",
                "technical": "bugs, errors, outages",
                "sales": "purchasing, upgrades",
            },
        )
        assert queue.choice in ["billing", "technical", "sales"]
        assert isinstance(queue.confidence, float)

        urgency = model.score(
            ticket,
            instructions="How urgent is this ticket?",
            criteria=["low", "medium", "high", "critical"],
        )
        assert 0.0 <= urgency.score <= 3.0

        churn = model.noul(ticket, instructions="Does the customer express churn intent?")
        assert 0.0 <= churn.noul <= 1.0

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
        batch = model.decide(ticket, questions)
        assert set(batch.answers.keys()) == {"queue", "urgency", "churn"}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_route():
    tmpdir = tempfile.mkdtemp()
    try:
        tiers = ["trivial", "standard", "complex", "reasoning"]
        criteria = {
            "trivial": "simple factual lookup with no ambiguity",
            "standard": "routine task requiring light reasoning",
            "complex": "multi-step problem or unclear requirements",
            "reasoning": "requires deep analysis or tool use",
        }
        examples = [
            System1Example(
                state={"query": "What is the capital of Australia?"},
                question_type="choice",
                instructions="Which model tier should handle this query?",
                criteria=criteria,
                answer={"choice": "trivial"},
                question_name="tier",
            ),
            System1Example(
                state={"query": "Refactor this module and add tests."},
                question_type="choice",
                instructions="Which model tier should handle this query?",
                criteria=criteria,
                answer={"choice": "complex"},
                question_name="tier",
            ),
        ]
        config = System1Config(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            classifier="LogisticRegression",
            output_dir=tmpdir,
        )
        model = DecisionSystem1(config=config)
        model.train(examples)
        router = DecisionRouter(tiers=tiers, criteria=criteria)
        result = model.route(
            {"query": "What is the capital of Australia?"},
            instructions="Which model tier should handle this query?",
            router=router,
        )
        assert result["tier"] in router.tiers
        assert isinstance(result["confidence"], float)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_train_decision_model_and_load(ticket_examples):
    tmpdir = tempfile.mkdtemp()
    try:
        path = train_decision_model(
            ticket_examples,
            config=System1Config(
                encoder_name="sentence-transformers/all-MiniLM-L6-v2",
                classifier="LogisticRegression",
                output_dir=tmpdir,
            ),
        )
        loaded = load_decision_model(path)
        assert loaded.predictor is not None
        result = loaded.choice(
            {"ticket": "Please refund my duplicate charge."},
            instructions="What queue should handle this ticket?",
            criteria={
                "billing": "refunds, charges, invoices",
                "technical": "bugs, errors, outages",
                "sales": "purchasing, upgrades",
            },
        )
        assert result.choice in ["billing", "technical", "sales"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_load_decision_examples_from_json(ticket_examples):
    tmpdir = tempfile.mkdtemp()
    try:
        records = [
            {
                "state": {"ticket": "I was charged twice, please refund."},
                "type": "choice",
                "instructions": "What queue should handle this ticket?",
                "criteria": {
                    "billing": "refunds, charges, invoices",
                    "technical": "bugs, errors, outages",
                    "sales": "purchasing, upgrades",
                },
                "answer": {"choice": "billing"},
                "name": "queue",
            },
            {
                "state": {"ticket": "The API returns 500 on every request."},
                "type": "choice",
                "instructions": "What queue should handle this ticket?",
                "criteria": {
                    "billing": "refunds, charges, invoices",
                    "technical": "bugs, errors, outages",
                    "sales": "purchasing, upgrades",
                },
                "answer": {"choice": "technical"},
                "name": "queue",
            },
        ]
        json_path = os.path.join(tmpdir, "examples.json")
        with open(json_path, "w") as f:
            json.dump(records, f)
        loaded = load_decision_examples(json_path)
        assert len(loaded) == 2
        assert loaded[0].question_name == "queue"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_module_level_with_env(ticket_examples):
    tmpdir = tempfile.mkdtemp()
    old_env = os.environ.get("NPCPY_DECISION_MODEL")
    try:
        model = DecisionSystem1(config=System1Config(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            classifier="LogisticRegression",
            output_dir=tmpdir,
        ))
        model.train(ticket_examples)
        os.environ["NPCPY_DECISION_MODEL"] = model.model_path

        ticket = {"ticket": "Please refund my duplicate charge."}
        result = decision_choice(
            ticket,
            instructions="What queue should handle this ticket?",
            criteria={
                "billing": "refunds, charges, invoices",
                "technical": "bugs, errors, outages",
                "sales": "purchasing, upgrades",
            },
        )
        assert result.choice in ["billing", "technical", "sales"]

        score_result = decision_score(
            ticket,
            instructions="How urgent is this ticket?",
            criteria=["low", "medium", "high", "critical"],
        )
        assert 0.0 <= score_result.score <= 3.0

        noul_result = decision_noul(ticket, instructions="Does the customer express churn intent?")
        assert 0.0 <= noul_result.noul <= 1.0

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
        ]
        predict_result = decision_predict(ticket, questions)
        assert "queue" in predict_result.answers
    finally:
        if old_env is None:
            os.environ.pop("NPCPY_DECISION_MODEL", None)
        else:
            os.environ["NPCPY_DECISION_MODEL"] = old_env
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_batch_decide(ticket_examples):
    tmpdir = tempfile.mkdtemp()
    try:
        model = DecisionSystem1(config=System1Config(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            classifier="LogisticRegression",
            output_dir=tmpdir,
        ))
        model.train(ticket_examples)
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
        ]
        batch = model.batch_decide(
            [
                {"ticket": "I was charged twice."},
                {"ticket": "The API is down."},
            ],
            questions,
        )
        assert len(batch.results) == 2
        assert batch.to_dict()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
