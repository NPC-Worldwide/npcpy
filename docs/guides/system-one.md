# System One Decisions

System One is npcpy's fast, low-latency decision layer. Instead of asking an LLM to
*generate* an answer, you ask a decision model to *choose* one, and you get back a
probability distribution over the options you supplied. That makes it a good fit for
ticket triage, model routing, content/safety classification, and any place where you
need a typed answer with a confidence score instead of prose.

There are two interchangeable backends, and they return the exact same result objects:

| Backend | How it works | Best for |
|---------|--------------|----------|
| no provider (default) | Trains a scikit-learn classifier over sentence-transformer embeddings of your labelled examples. Runs entirely on your machine with no model server. Pass `model_path=` to load one. | Your own label taxonomy and domain-specific decisions. |
| `"ollama"` | Sends `state` + `questions` to a local Ollama decision model via `POST /v1/systemone` (Ollama >= 0.35.0, based on TypeSafe's Jev API). No training required. | General-purpose decisions out of the box, near-instant latency, no API key. |
| `"gemini"` / `"perplexity"` / `"openai"` | Calls the hosted Decisions API of Google, Perplexity, or OpenAI over `POST /decisions`. No training required. | Multimodal input (PDF, audio, video, images) and managed infrastructure. |

All backends answer the same three question types:

- **`choice`** — pick one label from a `criteria` mapping of `label -> description`.
- **`noul`** — answer a yes/no question; returns `P(true)` as a float in `[0, 1]`. (`noul` is "no/yes" as a single scalar.)
- **`score`** — score the state against an ordered rubric of `criteria` levels.

## Ollama Backend (no training)

### Setup

Install Ollama 0.35 or newer, then pull a decision model.

```bash
ollama update          # or reinstall; System One requires >= 0.35.0
ollama pull tev1:0.8b   # small and quick
# or: ollama pull nimble   # 9B, from Bespoke Labs
```

Verify availability before you wire it into anything:

```python
from npcpy.gen.systemone import available, require_available

available(model="tev1:0.8b")
# {'available': True, 'reachable': True, 'version': '0.35.1', 'version_ok': True,
#  'model': 'tev1:0.8b', 'model_present': True, 'base_url': 'http://localhost:11434',
#  'error': None}

require_available(model="tev1:0.8b")   # raises with a fix-it hint if anything is missing
```

### Choosing, scoring, and yes/no

```python
from npcpy.gen.systemone import OllamaSystem1Client, OllamaSystem1Config

client = OllamaSystem1Client(OllamaSystem1Config(model="tev1:0.8b"))

ticket = {"ticket": "I was charged twice. Please refund the extra payment."}

team = client.choice(
    ticket,
    instructions="Which team should handle this ticket?",
    criteria={
        "billing": "Payments and refunds",
        "technical": "Bugs and integrations",
        "other": "None of the above",
    },
)
print(team.choice, team.confidence, team.probabilities)
# billing 0.8906 {'billing': 0.9781, 'technical': 0.0109, 'other': 0.011}

refund = client.noul(ticket, instructions="Does the customer explicitly ask for a refund?")
print(refund.noul)   # 0.997   -> P(true)

urgency = client.score(
    ticket,
    instructions="How urgent is this ticket?",
    criteria=["Routine", "Soon", "Urgent"],
)
print(urgency.score, urgency.legend)
# 0.815 {'0': 'Routine', '1': 'Soon', '2': 'Urgent'}
```

Note the `legend` on score results: it maps the probability keys (`"0"`, `"1"`, ...) back
to the rubric labels you sent, so you can render the distribution without tracking the
ordering yourself. `legend` is optional and is `None` on the local backend.

### Batch: many questions, one request

The endpoint scores every question against the full state in a **single** request.
Answers are not chained — each question is scored independently.

```python
result = client.predict(
    ticket,
    {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Payments and refunds",
                "technical": "Bugs and integrations",
                "other": "None of the above",
            },
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
    },
)

result.answers["team"]      # ChoiceResult(choice='billing', ...)
result.answers["refund"]    # NoulResult(noul=0.997)
result.answers["urgency"]   # ScoreResult(score=0.815, legend={...})
```

### Raw requests

If you want the untouched payload — for example to log token usage — call `systemone`
directly. It posts exactly the documented body (`model`, `state`, `questions`, optional
`keep_alive`) and returns the parsed JSON.

```python
from npcpy.gen.systemone import systemone

payload = systemone(
    {"ticket": "Our checkout has returned 500 errors since 9am."},
    {
        "label": {
            "type": "choice",
            "instructions": "Which label fits this ticket?",
            "criteria": {
                "billing": "Payments and refunds",
                "bug": "Software errors",
                "account": "Login and account access",
            },
        }
    },
    model="tev1:0.8b",
)
print(payload["answers"]["label"]["choice"], payload["usage"])
```

`state` may be a string, an object, or an array. Objects and arrays are sent as JSON
verbatim; anything else is JSON-encoded. It is not interpreted as chat messages.

## Hosted Decisions APIs

OpenAI, Gemini, Perplexity, TypeSafe, Cloudflare, Databricks, Fireworks, OpenRouter,
and self-hosted vLLM/Laya/Bespoke all expose decision-model endpoints. They take the
same concept as the Ollama endpoint -- shared evidence plus named questions, typed answers
with probabilities, no generated text. `npcpy` normalizes all of them onto a single
flat function: `get_decision_response`.

| Provider | Example model | Key env var |
|----------|---------------|-------------|
| `"openai"` | `gpt-6-luna` | `OPENAI_API_KEY` |
| `"gemini"` | `gemini-decision-1` | `GEMINI_API_KEY` |
| `"perplexity"` | `pplx-decider-latest` | `PERPLEXITY_API_KEY` |
| `"typesafe"` | `jev-latest` | `TYPESAFE_API_KEY` |
| `"ollama"` | `tev1:0.8b` | none (local) |

Use the public entrypoint exactly like `get_llm_response`:



You can also call the lower-level helpers directly:



All supported providers return the same plain dict shape: `{type, choice|noul|score,
confidence, probabilities}`.

### Hosted credentials

The only setting read from the environment is the API key, using the same
variables the rest of `npcpy` uses: `OPENAI_API_KEY`, `GEMINI_API_KEY`, and
`PERPLEXITY_API_KEY`. Provider and model are always passed explicitly.

## Local Backend (train your own)

Install the optional dependencies and label some examples.

```bash
pip install -e ".[system1]"
```

```python
from npcpy.ft.system1 import System1Config, System1Example, train_system1, load_system1

examples = [
    System1Example(
        state={"ticket": "I was charged twice, please refund."},
        question_type="choice",
        instructions="What queue should handle this ticket?",
        criteria={"billing": "refunds, charges, invoices",
                  "technical": "bugs, errors, outages",
                  "sales": "purchasing, upgrades"},
        answer={"choice": "billing"},
        question_name="queue",
    ),
    System1Example(
        state={"ticket": "The API returns 500 on every request."},
        question_type="choice",
        instructions="What queue should handle this ticket?",
        criteria={"billing": "refunds, charges, invoices",
                  "technical": "bugs, errors, outages",
                  "sales": "purchasing, upgrades"},
        answer={"choice": "technical"},
        question_name="queue",
    ),
    # ... plus "urgency" (score) and "churn" (noul) examples
]

config = System1Config(
    encoder_name="sentence-transformers/all-MiniLM-L6-v2",
    classifier="LogisticRegression",
    output_dir="./system1_ticket_model",
)
model_path = train_system1(examples, config)

predictor = load_system1(model_path)
result = predictor.choice(
    {"ticket": "Please refund the duplicate charge on my account."},
    instructions="What queue should handle this ticket?",
    criteria={"billing": "refunds, charges, invoices",
              "technical": "bugs, errors, outages",
              "sales": "purchasing, upgrades"},
    question_name="queue",
)
print(result.choice, result.confidence)
```

Training writes `classifiers.pkl`, `label_maps.json`, `meta.json`, and the saved encoder
into `output_dir`. `label_maps` records the labels each question was trained on; the
router uses it to check whether a trained model actually covers the labels you asked for.

### System1Config

| Field | Default | Description |
|-------|---------|-------------|
| `encoder_name` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-transformer used to embed states. |
| `classifier` | `LogisticRegression` | Any of `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `SVC`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `GaussianNB`, `MLPClassifier`. |
| `classifier_kwargs` | `{"max_iter": 1000}` | Keyword args passed to the classifier. |
| `output_dir` | `./system1_model` | Where the trained artifacts are written. |
| `device` | `cpu` | Encoder device. |

## The `gen.decision` Interface

`npcpy.gen.decision` is the unified, class-free surface over both the hosted and
local backends. Pick a backend per call with `provider=` and `model=`.



### Backend selection

The `provider=` argument takes the same values `get_llm_response` accepts:
`"ollama"`, `"openai"`, `"gemini"`, `"perplexity"`, `"typesafe"`, etc.
Aliases `"systemone"`, `"system_one"`, and `"jev"` are accepted for Ollama.

No `provider=` means you must pass `model_path=` pointing to a locally trained
`npcpy.ft.system1` model, or set `NPCPY_SYSTEM1_MODEL`.

### Environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `NPCPY_SYSTEM1_OLLAMA_MODEL` | Ollama client | Default decision model (e.g. `tev1:0.8b`). |
| `NPCPY_OLLAMA_BASE_URL` / `OLLAMA_HOST` | Ollama client | Ollama host; default `http://localhost:11434`. |
| `NPCPY_SYSTEM1_MODEL` | local backend | Path to a trained model directory. |

### Module-level helpers

| Function | Returns |
|----------|---------|
| `decision_choice(state, instructions, criteria, ...)` | dict with `choice`, `confidence`, `probabilities` |
| `decision_score(state, instructions, criteria, ...)` | dict with `score`, `confidence`, `probabilities`, `legend` |
| `decision_noul(state, instructions, ...)` | dict with `noul`, `confidence`, `probabilities` |
| `decision_predict(state, questions, ...)` | dict answer for the first question in the list |
| `decision_route(state, instructions, tiers, criteria=None, threshold=None, ...)` | dict with `tier`, `confidence`, `probabilities`, `tiers` |

Each accepts `model_path=`, `provider=`, `model=`, `api_url=`, and `api_key=`. A
hosted provider requires `model=`.

## Routing

`decision_route` turns a decision into a dispatch. Give it ordered `tiers` and
descriptions, and it returns the selected tier plus the confidence behind the choice.
`threshold` is a floor: if the winning confidence is below it, routing falls back to the
**last** tier, which you should order as your safest/most capable option.



With the Ollama backend, criteria are resolved per call, so the router uses the model
directly. With a trained model, the router checks the model's saved label maps first and
falls back to the LLM path if the trained labels don't cover the tiers you asked for —
that way a stale or narrow model degrades gracefully instead of guessing.

## Over HTTP

The Flask server exposes the same decisions. `provider`, `model`, and `api_url` are
accepted per request on all of these endpoints:

| Endpoint | Body fields |
|----------|-------------|
| `POST /api/system1/predict` | `state`, `questions`, `provider`, `model`, `api_url`, `model_path` |
| `POST /api/system1/choice` | `state`, `instructions`, `criteria`, ... |
| `POST /api/system1/noul` | `state`, `instructions`, ... |
| `POST /api/system1/score` | `state`, `instructions`, `criteria`, ... |
| `POST /api/decision/predict` | `state`, `questions` (list), `provider`, `model`, `api_url`, `model_path` |
| `POST /api/decision/choice` | `state`, `instructions`, `criteria`, ... |
| `POST /api/decision/noul` | `state`, `instructions`, ... |
| `POST /api/decision/score` | `state`, `instructions`, `criteria`, ... |

```bash
curl -s localhost:5337/api/system1/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "state": {"ticket": "I was charged twice. Please refund the extra payment."},
    "questions": {
      "team": {"type": "choice",
               "instructions": "Which team should handle this ticket?",
               "criteria": {"billing": "Payments and refunds",
                            "technical": "Bugs and integrations",
                            "other": "None of the above"}},
      "refund": {"type": "noul",
                 "instructions": "Does the customer explicitly ask for a refund?"}
    },
    "backend": "ollama",
    "model": "tev1:0.8b"
  }'
```

Omitting `backend` keeps the previous behaviour: the locally trained model, if
`model_path` or `NPCPY_SYSTEM1_MODEL` is set. See [Serving & Deployment](serving.md).

## In Jinxes

The compiler exposes the System One helpers to jinx steps, so a workflow can branch on a
decision without leaving the template:

```jinja
{% set steps = [
  {"name": "triage",
   "code": "team = systemone(
       context['ticket'],
       {'team': {'type': 'choice',
                 'instructions': 'Which team should handle this ticket?',
                 'criteria': {'billing': 'Payments and refunds',
                              'technical': 'Bugs and integrations'}}},
       model='tev1:0.8b')
   context['route'] = team['answers']['team']['choice']
   output = str(context['route'])"},
] %}
```

Available globals: `systemone`, `load_ollama_system1`, `OllamaSystem1Client`,
`OllamaSystem1Config`, `systemone_available`, `decision_choice`, `decision_score`,
`decision_noul`, `decision_predict`, `decision_route`, `get_decision_response`.

## Failure Modes

`OllamaSystem1Error` carries an actionable message rather than a bare status code:

| Situation | Message points at |
|-----------|-------------------|
| 404 from the endpoint | Ollama is older than 0.35.0 — upgrade, then `ollama pull <model>`. |
| 413 | The 64 KiB request limit; shorten the state or drop questions. |
| 400 | The model isn't a decision model, or the prompt exceeds the context window. |
| Connection refused | `ollama serve`, plus host/port confirmation. |
| Request body > 64 KiB | Caught client-side before the round trip. |

The 64 KiB limit is enforced locally as well as surfaced from the server, so oversized
payloads fail fast. Prompts are never truncated by Ollama: each rendered prompt must fit
the loaded context window with two token positions to spare. Streaming, images, tools,
and generation controls are not supported by this endpoint — it returns one JSON response.

## Notes

- Ollama decides one request at a time; `predict` batches your questions into a single
  POST, not one POST per question.
- `keep_alive` is passed through (`"5m"` by default, matching Ollama's own default). `0`
  unloads the model after the request; a negative value keeps it resident.
- The Ollama backend does not support `train()` or `load(path)`. Both raise `ValueError`
  explaining that the models are pre-trained and pulled rather than fitted.
- `ScoreResult.legend` is additive: `to_dict()` only includes it when it is set, so
  existing consumers are unaffected.
- The mock server in `tests/ollama_mock.py` reproduces this API, so you can test against
  the full request/response contract without installing Ollama. See
  `examples/example_system1_ollama.py` for a runnable end-to-end example.
