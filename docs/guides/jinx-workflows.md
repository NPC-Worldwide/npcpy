# Jinx Workflows

Jinxs (Jinja Execution templates) are prompt-based workflow templates that chain multiple steps together. Each step can use natural language processing or Python code execution, and steps can reference prior results through Jinja templating. Because jinxs are prompt-driven, they work with any model -- including those without tool-calling support.

## What is a Jinx

A jinx is a YAML-defined workflow with named steps. Each step has an engine (`natural` for LLM processing, `python` for code execution) and a code block that can reference inputs and previous step outputs using `{{ variable }}` syntax.

Jinxs solve the problem of composing multi-step LLM workflows declaratively. Instead of writing procedural code to chain prompts, you define a template that npcpy executes step by step, threading context between stages.

## Two Engines

### Natural Engine

Steps with `engine: "natural"` send the rendered code block as a prompt to the NPC's LLM. The response is stored in the context and available to later steps.

### Python Engine

Steps with `engine: "python"` execute the rendered code block as Python. The `context` dict is available for reading and writing state. Set `output` to pass a result string to subsequent steps. Common libraries (`os`, `json`, `pandas`, `numpy`, `re`, `subprocess`, `pathlib`) are pre-imported in the execution environment.

## YAML File Format

Jinx files use the `.jinx` extension and follow this structure:

```yaml
jinx_name: "my_workflow"
description: "What this workflow does"
inputs:
  - "input_one"
  - "input_two"
steps:
  - name: "step_one"
    engine: "python"
    code: |
      # Python code here
      # Access inputs via {{ input_one }} or context['input_one']
      output = "result from step one"

  - name: "step_two"
    engine: "natural"
    code: |
      Based on the previous step: {{ step_one }}
      Do something with {{ input_two }}.
```

Key fields:

- `jinx_name` -- identifier used to reference the jinx
- `description` -- human-readable summary
- `inputs` -- list of required input parameter names
- `steps` -- ordered list of step definitions, each with `name`, `engine`, and `code`

## Step References with Jinja Templating

Steps reference inputs and prior step outputs using `{{ variable_name }}` syntax. When a step completes, its output is stored in the context under its `name` and becomes available to all subsequent steps.

```yaml
steps:
  - name: "load_data"
    engine: "python"
    code: |
      import pandas as pd
      df = pd.read_csv('{{ file_path }}')
      context['row_count'] = len(df)
      output = f"Loaded {len(df)} rows"

  - name: "analyze"
    engine: "natural"
    code: |
      The dataset at {{ file_path }} has {{ row_count }} rows.
      Previous step said: {{ load_data }}
      Provide analysis and insights.
```

## Simple Jinx: Data Analyzer

Create a file called `data_analyzer.jinx`:

```yaml
jinx_name: "data_analyzer"
description: "Analyze CSV data and generate insights"
inputs:
  - "file_path"
  - "analysis_type"
steps:
  - name: "load_data"
    engine: "python"
    code: |
      import pandas as pd
      import numpy as np

      df = pd.read_csv('{{ file_path }}')
      print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

      context['dataframe'] = df
      context['row_count'] = len(df)
      context['column_count'] = len(df.columns)
      output = f"Loaded {len(df)} rows, {len(df.columns)} columns"

  - name: "analyze_data"
    engine: "python"
    code: |
      df = context['dataframe']
      analysis_type = '{{ analysis_type }}'.lower()

      if analysis_type == 'basic':
          stats = df.describe()
          context['statistics'] = stats.to_dict()
          output = f"Basic statistics computed for {len(df.columns)} columns"
      elif analysis_type == 'correlation':
          numeric_df = df.select_dtypes(include=[np.number])
          if len(numeric_df.columns) > 1:
              corr_matrix = numeric_df.corr()
              context['correlation_matrix'] = corr_matrix.to_dict()
              output = f"Correlation matrix computed for {len(numeric_df.columns)} numeric columns"
          else:
              output = "Not enough numeric columns for correlation analysis"
      else:
          output = f"Unknown analysis type: {analysis_type}"

  - name: "generate_report"
    engine: "natural"
    code: |
      Based on the data analysis results:

      - Dataset has {{ row_count }} rows and {{ column_count }} columns
      - Analysis type: {{ analysis_type }}

      {% if statistics %}
      Key statistics: {{ statistics }}
      {% endif %}

      {% if correlation_matrix %}
      Correlation insights: {{ correlation_matrix }}
      {% endif %}

      Generate a comprehensive summary report of the key findings and insights.
```

## Complex Jinx: Research Pipeline

Create `research_pipeline.jinx`:

```yaml
jinx_name: "research_pipeline"
description: "Research a topic, analyze sources, and generate a report"
inputs:
  - "research_topic"
  - "output_format"
steps:
  - name: "gather_info"
    engine: "natural"
    code: |
      Research the topic: {{ research_topic }}

      Provide comprehensive information including:
      1. Key concepts and definitions
      2. Current trends and developments
      3. Major challenges or controversies
      4. Future outlook

      Focus on recent, credible sources and provide specific examples.

  - name: "analyze_findings"
    engine: "python"
    code: |
      import re
      from collections import Counter

      research_text = context.get('llm_response', '')

      sentences = re.split(r'[.!?]', research_text)
      context['sentence_count'] = len([s for s in sentences if len(s.strip()) > 10])

      words = re.findall(r'\b[A-Z][a-z]+\b', research_text)
      common_terms = Counter(words).most_common(10)
      context['key_terms'] = dict(common_terms)

      output = f"Analysis complete: {context['sentence_count']} sentences, " \
               f"top terms: {list(context['key_terms'].keys())[:5]}"

  - name: "format_report"
    engine: "natural"
    code: |
      Based on the research findings about {{ research_topic }}, create a
      well-structured report in {{ output_format }} format.

      Research Summary:
      {{ gather_info }}

      Key Statistics:
      - Number of key points covered: {{ sentence_count }}
      - Most mentioned terms: {{ key_terms }}

      Format this as a professional {{ output_format }} with:
      1. Executive Summary
      2. Main Findings
      3. Analysis and Insights
      4. Recommendations
      5. Conclusion
```

## Using Jinxs with NPCs

Reference jinx files by name when creating an NPC, then call `execute_jinx()`.

```python
from npcpy.npc_compiler import NPC

data_scientist = NPC(
    name='Data Scientist',
    primary_directive='You are an expert data scientist specializing in data analysis.',
    jinxs=['data_analyzer'],  # references the jinx file by name
    model='llama3.2',
    provider='ollama'
)

result = data_scientist.execute_jinx(
    'data_analyzer',
    {
        'file_path': './sales_data.csv',
        'analysis_type': 'basic'
    }
)
print(result['output'])
```

## Creating Jinx Objects Programmatically

Instead of YAML files, you can build jinxs in Python using the `Jinx` class with a `jinx_data` dict.

```python
from npcpy.npc_compiler import Jinx

file_reader_jinx = Jinx(jinx_data={
    "jinx_name": "file_reader",
    "description": "Read a file and summarize its contents",
    "inputs": ["filename"],
    "steps": [
        {
            "name": "read_file",
            "engine": "python",
            "code": """
import os
with open(os.path.abspath('{{ filename }}'), 'r') as f:
    content = f.read()
output = content
            """
        },
        {
            "name": "summarize_content",
            "engine": "natural",
            "code": """
                Summarize the content of the file: {{ read_file }}.
            """
        }
    ]
})
```

You can also load from a YAML file path:

```python
research_jinx = Jinx(jinx_path='./research_pipeline.jinx')
```

## Executing Jinxs Directly

Call `execute()` on a Jinx object, passing input values and an NPC to handle the natural-language steps.

```python
from npcpy.npc_compiler import Jinx, NPC

research_jinx = Jinx(jinx_path='./research_pipeline.jinx')

npc = NPC(
    name='Research Assistant',
    primary_directive='You are a research assistant specialized in analysis and reporting.',
    model='gemma3:4b',
    provider='ollama'
)

result = research_jinx.execute(
    input_values={
        'research_topic': 'artificial intelligence in healthcare',
        'output_format': 'markdown'
    },
    npc=npc
)
print(result['output'])
```

The `execute()` method returns the full context dict, which includes all step outputs, the final `output`, and any values stored in `context` during Python steps.

## Jinx Composition

Jinxs can call other jinxs by referencing them through Jinja templating when used within a team or NPC that has multiple jinxs loaded. Step outputs from one jinx can feed into another through shared context.

```python
from npcpy.npc_compiler import NPC, Team, Jinx

# A jinx that reads and processes files
file_processor = Jinx(jinx_data={
    "jinx_name": "file_processor",
    "description": "Read and process a file",
    "inputs": ["filepath"],
    "steps": [
        {
            "name": "read",
            "engine": "python",
            "code": """
with open('{{ filepath }}', 'r') as f:
    content = f.read()
context['file_content'] = content
output = content[:500]
            """
        }
    ]
})

# A jinx that summarizes content
summarizer = Jinx(jinx_data={
    "jinx_name": "summarizer",
    "description": "Summarize provided content",
    "inputs": ["content"],
    "steps": [
        {
            "name": "summarize",
            "engine": "natural",
            "code": "Summarize the following content concisely: {{ content }}"
        }
    ]
})

# An NPC with both jinxs available
analyst = NPC(
    name='Analyst',
    primary_directive='You analyze and summarize documents.',
    model='llama3.2',
    provider='ollama',
)

# Create a team with shared jinxs
team = Team(
    npcs=[analyst],
    forenpc=analyst,
    jinxs=[file_processor, summarizer]
)

# Execute individual jinxs through the NPC
result = analyst.execute_jinx('file_processor', {'filepath': './data.txt'})
print(result['output'])
```

Team-level jinxs are rendered during team initialization and distributed to all member NPCs, so any NPC in the team can execute any team jinx. This enables the forenpc to delegate jinx-based workflows to the most appropriate team member.

## Skills: Knowledge-Content Jinxs

Skills are jinxs that serve instructional content instead of executing code. They use the `skill.jinx` sub-jinx (just like code jinxs use `python.jinx` or `sh.jinx`) and return sections of knowledge on demand.

You can author skills as `SKILL.md` folders or as `.jinx` files with `engine: skill` steps. Either way, they end up in the same `jinxs_dict` and are assigned to agents through the same `jinxs:` list in `.npc` files.

```yaml
# reviewer.npc
jinxs:
  - lib/core/sh
  - lib/core/python
  - skills/code-review
  - skills/debugging
```

The agent calls `code-review(section=correctness)` the same way it calls `sh(command=ls)` — through the same jinx pipeline. See the [Skills guide](skills.md) for full details on authoring and usage.
