---
name: natural
description: Render the provided prompt template with Jinja context and send it to
  the active NPC's LLM. The rendered text becomes the user prompt; the LLM response
  is placed in context['output'].
---

# natural

Render the provided prompt template with Jinja context and send it to the active NPC's LLM. The rendered text becomes the user prompt; the LLM response is placed in context['output'].

## Inputs

- `prompt`
- `system` (default: `''`)

## Steps

- `render_and_send` → [`render_and_send.py`](./render_and_send.py)

## Usage

```
/run_jinx jinx_ref=natural input_values={"prompt": "<value>", "system": ""}
```
