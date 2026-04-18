---
name: create_jinx
description: Write a new .jinx file from inputs (name, description, inputs spec as
  JSON list of strings, python body). Agents use this mid-task to crystallize a repeated
  block of code into a reusable jinx. The jinx is written to the team's jinxes/lib
  dir by default and registered with the running team so it is immediately callable
  without a restart.
---

# create_jinx

Write a new .jinx file from inputs (name, description, inputs spec as JSON list of strings, python body). Agents use this mid-task to crystallize a repeated block of code into a reusable jinx. The jinx is written to the team's jinxes/lib dir by default and registered with the running team so it is immediately callable without a restart.

## Inputs

- `jinx_name`
- `description`
- `inputs_spec` (default: `'[]'`)
- `python_code`
- `target_subdir` (default: `'lib'`)

## Steps

- `write_jinx` → [`write_jinx.py`](./write_jinx.py)

## Usage

```
/run_jinx jinx_ref=create_jinx input_values={"jinx_name": "<value>", "description": "<value>", "inputs_spec": "[]", "python_code": "<value>", "target_subdir": "lib"}
```
