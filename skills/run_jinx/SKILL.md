---
name: run_jinx
description: Execute a jinx by name (already loaded on the team) or by filesystem
  path (e.g. a freshly created_jinx that isn't yet registered), passing input values
  as a JSON object. Returns the jinx's output field or the full context dict. Use
  this to run a jinx you just wrote without exiting the session.
---

# run_jinx

Execute a jinx by name (already loaded on the team) or by filesystem path (e.g. a freshly created_jinx that isn't yet registered), passing input values as a JSON object. Returns the jinx's output field or the full context dict. Use this to run a jinx you just wrote without exiting the session.

## Inputs

- `jinx_ref`
- `input_values` (default: `'{}'`)

## Steps

- `execute` → [`execute.py`](./execute.py)

## Usage

```
/run_jinx jinx_ref=run_jinx input_values={"jinx_ref": "<value>", "input_values": "{}"}
```
