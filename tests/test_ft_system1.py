import os
import tempfile
import shutil

import pytest

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


def test_train_and_predict(ticket_examples):
    tmpdir = tempfile.mkdtemp()
    try:
        config = System1Config(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            classifier="LogisticRegression",
            output_dir=tmpdir,
        )
        path = train_system1(ticket_examples, config)
        assert os.path.isdir(path)
        predictor = load_system1(path)

        ticket = {"ticket": "The database is down and users cannot log in."}
        queue = predictor.choice(
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
        assert "billing" in queue.probabilities

        urgency = predictor.score(
            ticket,
            instructions="How urgent is this ticket?",
            criteria=["low", "medium", "high", "critical"],
        )
        assert 0.0 <= urgency.score <= 3.0

        churn = predictor.noul(ticket, instructions="Does the customer express churn intent?")
        assert 0.0 <= churn.noul <= 1.0

        batch = predictor.predict(
            ticket,
            questions={
                "queue": {
                    "type": "choice",
                    "instructions": "What queue should handle this ticket?",
                    "criteria": {
                        "billing": "refunds, charges, invoices",
                        "technical": "bugs, errors, outages",
                        "sales": "purchasing, upgrades",
                    },
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent is this ticket?",
                    "criteria": ["low", "medium", "high", "critical"],
                },
                "churn": {
                    "type": "noul",
                    "instructions": "Does the customer express churn intent?",
                },
            },
        )
        assert set(batch.answers.keys()) == {"queue", "urgency", "churn"}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_module_level_with_env(ticket_examples):
    tmpdir = tempfile.mkdtemp()
    old_env = os.environ.get("NPCPY_SYSTEM1_MODEL")
    try:
        config = System1Config(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            classifier="LogisticRegression",
            output_dir=tmpdir,
        )
        train_system1(ticket_examples, config)
        os.environ["NPCPY_SYSTEM1_MODEL"] = tmpdir

        ticket = {"ticket": "Please refund my duplicate charge."}
        result = choice(
            ticket,
            instructions="What queue should handle this ticket?",
            criteria={
                "billing": "refunds, charges, invoices",
                "technical": "bugs, errors, outages",
                "sales": "purchasing, upgrades",
            },
        )
        assert result.choice in ["billing", "technical", "sales"]
    finally:
        if old_env is None:
            os.environ.pop("NPCPY_SYSTEM1_MODEL", None)
        else:
            os.environ["NPCPY_SYSTEM1_MODEL"] = old_env
        shutil.rmtree(tmpdir, ignore_errors=True)
