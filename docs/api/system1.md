# System One

Fast, typed decisions (choice / yes-no / scoring) from either a locally trained
classifier over embeddings or a local Ollama decision model.

See the [System One Decisions guide](../guides/system-one.md) for usage.

## npcpy.ft.system1

Local backend: train a small classifier over sentence-transformer embeddings.

::: npcpy.ft.system1
    options:
      show_source: true
      members: true
      filters:
        - "!^_"
        - "!^[A-Z]{2,}"
        - "!test_"
      inherited_members: false
      show_root_heading: false
      show_if_no_docstring: true

## npcpy.gen.systemone

Ollama backend: `POST /v1/systemone` against a local decision model.

::: npcpy.gen.systemone
    options:
      show_source: true
      members: true
      filters:
        - "!^_"
        - "!^[A-Z]{2,}"
        - "!test_"
      inherited_members: false
      show_root_heading: false
      show_if_no_docstring: true

## npcpy.gen.decision

Unified decision surface with backend selection, batching, and routing.

::: npcpy.gen.decision
    options:
      show_source: true
      members: true
      filters:
        - "!^_"
        - "!^[A-Z]{2,}"
        - "!test_"
      inherited_members: false
      show_root_heading: false
      show_if_no_docstring: true
