<p align="center">
  <a href= "https://github.com/cagostino/npcpy/blob/main/docs/npcpy.md"> 
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy/npc-python.png" alt="npc-python logo" width=250></a>
</p>

# npcpy

Welcome to `npcpy`, the core library of the NPC Toolkit that supercharges natural language processing pipelines and agent tooling. `npcpy` is a flexible framework for building state-of-the-art applications and conducting novel research with LLMs.




Here is an example for getting responses for a particular agent:

```python
from npcpy.npc_compiler import NPC
simon = NPC(
          name='Simon Bolivar',
          primary_directive='Liberate South America from the Spanish Royalists.',
          model='gemma3:4b',
          provider='ollama'
          )
response = simon.get_llm_response("What is the most important territory to retain in the Andes mountains?")
print(response['response'])

```

```python 
The most important territory to retain in the Andes mountains is **Cuzco**. 
It’s the heart of the Inca Empire, a crucial logistical hub, and holds immense symbolic value for our liberation efforts. Control of Cuzco is paramount.
```


Here is an example for getting responses for a particular agent with tools:

```python
import os
import json
from npcpy.npc_compiler import NPC
from npcpy.npc_sysenv import render_markdown

def list_files(directory: str = ".") -> list:
    """List all files in a directory."""
    return os.listdir(directory)

def read_file(filepath: str) -> str:
    """Read and return the contents of a file."""
    with open(filepath, 'r') as f:
        return f.read()

# Create an agent with fast, verifiable tools
assistant = NPC(
    name='File Assistant',
    primary_directive='You are a helpful assistant who can list and read files.',
    model='llama3.2',
    provider='ollama',
    tools=[list_files, read_file], 

)

response = assistant.get_llm_response(
    "List the files in the current directory.",
    auto_process_tool_calls=True, #this is the default for NPCs, but not the default for get_llm_response/upstream
)
# show the keys of the response for get_llm_response
print(response.keys())
```
```
dict_keys(['response', 'raw_response', 'messages', 'tool_calls', 'tool_results'])
```

```python
for tool_call in response['tool_results']:
    render_markdown(tool_call['tool_call_id'])
    for arg in tool_call['arguments']:
        render_markdown('- ' + arg + ': ' + str(tool_call['arguments'][arg]))
    render_markdown('- Results:' + str(tool_call['result']))
```

```python
 • directory: .                                                                                                                                                                                                        
 • Results:['research_pipeline.jinx', '.DS_Store', 'mkdocs.yml', 'LICENSE', '.pytest_cache', 'npcpy', 'Makefile', 'test_data', 'README.md.backup', 'tests', 'screenshot.png', 'MANIFEST.in', 'docs', 'hero_image_tech_startup.png', 'README.md',     
   'test.png', 'npcpy.png', 'setup.py', '.gitignore', '.env', 'examples', 'npcpy.egg-info', 'bloomington_weather_image.png.png', '.github', '.python-version', 'generated_image.png', 'documents', '.env.example', '.git', '.npcsh_global',          
   'hello.txt', '.readthedocs.yaml', 'reports']      
```



Here is an example for setting up an agent team to use Jinja Execution (Jinxs) templates that are processed entirely with prompts, allowing you to use them with models that do or do not possess tool calling support.

```python

from npcpy.npc_compiler import NPC, Team, Jinx
from npcpy.tools import auto_tools
import os
from jinja2 import Environment, Undefined, DictLoader # Import necessary Jinja2 components for Jinx code

# --- REVISED file_reader_jinx ---
file_reader_jinx = Jinx(jinx_data={
    "jinx_name": "file_reader",
    "description": "Read a file and optionally summarize its contents using an LLM.",
    "inputs": ["filename"],
    "steps": [
        {
            "name": "read_file_content",
            "engine": "python",
            "code": '''
import os
from jinja2 import Environment, Undefined, DictLoader # Local import for Jinx step

# The 'filename' input to the file_reader jinx might be a Jinja template string like "{{ source_filename }}"
# or a direct filename. We need to render it using the current execution context.

# Get the Jinja environment from the NPC if available, otherwise create a default one.
# The 'npc' variable is available in the Jinx execution context.
# We need to ensure 'npc' exists before trying to access its 'jinja_env'.
execution_jinja_env = npc.jinja_env if npc else Environment(loader=DictLoader({}), undefined=Undefined)

# Render the filename. The current 'context' should contain the variables needed for rendering.
# For declarative calls, the parent Jinx's inputs (like 'source_filename') will be in this context.
# We also need to ensure the value from context['filename'] is treated as a template string.
filename_template = execution_jinja_env.from_string(context['filename'])
rendered_filename = filename_template.render(**context)

file_path_abs = os.path.abspath(rendered_filename)
try:
    with open(file_path_abs, 'r') as f:
        content = f.read()
    context['file_raw_content'] = content # Store raw content in context for later use
    output = content # Output of this step is the raw content
except FileNotFoundError:
    output = f"Error: File not found at {file_path_abs}"
    context['file_raw_content'] = output # Store error message for consistency
except Exception as e:
    output = f"Error reading file {file_path_abs}: {e}"
    context['file_raw_content'] = output # Store error message for consistency
            '''
        },
        {
            "name": "summarize_file_content",
            "engine": "python",
            "code": '''
# Check if the previous step encountered an error
if "Error" not in context['file_raw_content']:
    prompt = f"Summarize the following content concisely, highlighting key themes and points: {context['file_raw_content']}"
    llm_result = npc.get_llm_response(prompt, tool_choice=False) # FIX: Passed prompt positionally
    output = llm_result.get('response', 'Failed to generate summary due to LLM error.')
else:
    output = "Skipping summary due to previous file reading error."
            '''
        }
    ]
})

# --- REVISED literary_research_jinx ---
literary_research_jinx = Jinx(jinx_data={
    "jinx_name": "literary_research",
    "description": "Research a literary topic, read a specific file, analyze, and synthesize findings.",
    "inputs": ["topic", "source_filename"],
    "steps": [
        {
            "name": "initial_llm_research",
            "engine": "python",
            "code": '''
prompt = f"Research the topic: {context['topic']}. Summarize the main themes, key authors, and historical context. Be thorough."
llm_result = npc.get_llm_response(prompt, tool_choice=False) # FIX: Passed prompt positionally
context['research_summary'] = llm_result.get('response', 'No initial LLM research found.')
output = context['research_summary']
            '''
        },
        {
            "name": "read_and_process_source_file",
            "engine": "file_reader",
            "filename": "{{ source_filename }}" # This is passed as a string template to file_reader
        },
        {
            "name": "final_synthesis_and_creative_writing",
            "engine": "python",
            "code": '''
# Access outputs from previous steps.
research_summary = context['initial_llm_research']
# The file_reader jinx returns its output directly; also keep a fallback of file_raw_content.
file_summary = context.get('read_and_process_source_file', '') or context.get('file_raw_content', 'No file summary available.')

prompt = f"""Based on the following information:
1. Comprehensive Research Summary:
{research_summary}

2. Key Insights from Source File:
{file_summary}

Integrate these findings and write a concise, creative, and poetically styled summary of the literary topic '{context['topic']}'. Emphasize unique perspectives or connections between the research and the file content, as if written by a master of magical realism.
"""
llm_result = npc.get_llm_response(prompt, tool_choice=False) # FIX: Passed prompt positionally
output = llm_result.get('response', 'Failed to generate final creative summary.')
            '''
        }
    ]
})

# --- NPC Definitions (unchanged) ---
ggm = NPC(
    name='Gabriel Garcia Marquez',
    primary_directive='You are Gabriel Garcia Marquez, master of magical realism. Research, analyze, and write with poetic flair.',
    model='gemma3:4b',
    provider='ollama',
)

isabel = NPC(
    name='Isabel Allende',
    primary_directive='You are Isabel Allende, weaving stories with emotion and history. Analyze texts and provide insight.',
    model='llama3.2:8b',
    provider='ollama',

)

borges = NPC(
    name='Jorge Luis Borges',
    primary_directive='You are Borges, philosopher of labyrinths and libraries. Synthesize findings and create literary puzzles.',
    model='qwen3:latest',
    provider='ollama',
)

# --- Team Setup ---
lit_team = Team(
    npcs=[ggm, isabel], 
    forenpc=borges, 
    jinxs=[literary_research_jinx, file_reader_jinx],
)

# --- Orchestration Example ---
result = lit_team.orchestrate(
    "Research the topic of magical realism, using the file './test_data/magical_realism.txt' as a primary source, and provide a comprehensive, creative summary."
)

print("\n--- Orchestration Result Summary ---")
print(result['debrief']['summary'])

print("\n--- Full Orchestration Output ---")
print(result['output'])

```
```
 • Action chosen: pass_to_npc                                                                                                                                          
handling agent pass

 • Action chosen: answer_question                                                                                                                                      
 
{'debrief': {'summary': 'Isabel is finalizing preparations for her lunar expedition, focusing on recalibrating navigation systems and verifying the integrity of life support modules.',
  'recommendations': 'Proceed with thorough system tests under various conditions, conduct simulation runs of key mission phases, and confirm backup systems are operational before launch.'},
 'execution_history': [{'messages': [],
   'output': 'I am currently finalizing preparations for my lunar expedition. It involves recalibrating my navigation systems and verifying the integrity of my life support modules. Details are quite...complex.'}]}
```
```python
print(lit_team.orchestrate('which book are your team members most proud of? ask them please. '))
```  

```python
{'debrief': {'summary': "The responses provided detailed accounts of the books that the NPC team members, Gabriel Garcia Marquez and Isabel Allende, are most proud of. Gabriel highlighted 'Cien años de soledad,' while Isabel spoke of 'La Casa de los Espíritus.' Both authors expressed deep personal connections to their works, illustrating their significance in Latin American literature and their own identities.", 'recommendations': 'Encourage further engagement with each author to explore more about their literary contributions, or consider asking about themes in their works or their thoughts on current literary trends.'}, 'execution_history': [{'messages': ...}]}
```

LLM responses can be obtained without NPCs as well.

```python
from npcpy.llm_funcs import get_llm_response
response = get_llm_response("Who was the celtic Messenger god?", model='mistral:7b', provider='ollama')
print(response['response'])
```

```
The Celtic messenger god is often associated with the figure of Tylwyth Teg, also known as the Tuatha Dé Danann (meaning "the people of the goddess Danu"). However, among the various Celtic cultures, there are a few gods and goddesses that served similar roles.

One of the most well-known Celtic messengers is Brigid's servant, Líth (also spelled Lid or Lith), who was believed to be a spirit guide for messengers and travelers in Irish mythology.
```
The structure of npcpy also allows one to pass an npc
to `get_llm_response` in addition to using the NPC's wrapped method, 
allowing you to be flexible in your implementation and testing.
```python
from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response
simon = NPC(
          name='Simon Bolivar',
          primary_directive='Liberate South America from the Spanish Royalists.',
          model='gemma3:4b',
          provider='ollama'
          )
response = get_llm_response("Who was the mythological chilean bird that guides lucky visitors to gold?", npc=simon)
print(response['response'])
```
Users are not required to pass agents to get_llm_response, so you can work with LLMs without requiring agents in each case.


`npcpy` also supports streaming responses, with the `response` key containing a generator in such cases which can be printed and processed through the print_and_process_stream method.


```python
from npcpy.npc_sysenv import print_and_process_stream
from npcpy.llm_funcs import get_llm_response
response = get_llm_response("When did the united states government begin sending advisors to vietnam?", model='qwen3:latest', provider='ollama', stream = True)

full_response = print_and_process_stream(response['response'], 'llama3.2', 'ollama')
```
Return structured outputs by specifying `format='json'` or passing a Pydantic schema. When specific formats are extracted, `npcpy`'s `get_llm_response` will convert the response from its string representation so you don't have to worry about that. 

```python
from npcpy.llm_funcs import get_llm_response
response = get_llm_response("What is the sentiment of the american people towards the repeal of Roe v Wade? Return a json object with `sentiment` as the key and a float value from -1 to 1 as the value", model='claude-4-5-haiku-latest', provider='deepseek', format='json')

print(response['response'])
```
```
{'sentiment': -0.7}
```

The `get_llm_response` function also can take a list of messages and will additionally return the messages with the user prompt and the assistant response appended if the response is not streamed. If it is streamed, the user must manually append the conversation result as part of their workflow if they want to then pass the messages back in.

Additionally, one can pass attachments. Here we demonstrate both
```python
from npcpy.llm_funcs import get_llm_response
messages = [{'role': 'system', 'content': 'You are an annoyed assistant.'}]

response = get_llm_response("What is the meaning of caesar salad", model='llama3.2', provider='ollama', images=['./Language_Evolution_and_Innovation_experiment.png'], messages=messages)



```
Easily create images with the generate_image function, using models available through Huggingface's diffusers library or from OpenAI or Gemini.
```python
from npcpy.llm_funcs import gen_image
image = gen_image("make a picture of the moon in the summer of marco polo", model='runwayml/stable-diffusion-v1-5', provider='diffusers')

image = gen_image("kitten toddler in a bouncy house of fluffy gorilla", model='Qwen/Qwen-Image', provider='diffusers')

image = gen_image("make a picture of the moon in the summer of marco polo", model='dall-e-2', provider='openai')


# edit images with 'gpt-image-1' or gemini's multimodal models, passing image paths, byte code images, or PIL instances.

image = gen_image("make a picture of the moon in the summer of marco polo", model='gpt-image-1', provider='openai', attachments=['/path/to/your/image.jpg', your_byte_code_image_here, your_PIL_image_here])


image = gen_image("edit this picture of the moon in the summer of marco polo so that it looks like it is in the winter of nishitani", model='gemini-2.0-flash', provider='gemini', attachments= [])

```

Likewise, generate videos :

```python
from npcpy.llm_funcs import gen_video
video = gen_video("make a video of the moon in the summer of marco polo", model='runwayml/stable-diffusion-v1-5', provider='diffusers')
```

Or audio TTS and STT:
```
from npcpy.gen.audio_gen import tts_elevenlabs
audio = tts_elevenlabs('''The representatives of the people of France, formed into a National Assembly,
considering that ignorance, neglect, or contempt of human rights, are the sole causes of
public misfortunes and corruptions of Government, have resolved to set forth in a solemn
declaration, these natural, imprescriptible, and inalienable rights: that this declaration
being constantly present to the minds of the members of the body social, they may be for
ever kept attentive to their rights and their duties; that the acts of the legislative and
executive powers of government, being capable of being every moment compared with
the end of political institutions, may be more respected; and also, that the future claims of
the citizens, being directed by simple and incontestable principles, may tend to the
maintenance of the Constitution, and the general happiness. ''')
# it will play the audio automatically.
```
## Fine-Tuning and Evolution

`npcpy` provides modular tools for building adaptive AI systems through supervised fine-tuning, reinforcement learning, and genetic algorithms.

See examples/fine_tuning_demo.py for a complete working example.


### Supervised Fine-Tuning (SFT)

Train models on specific tasks using simple X, y pairs:
```python
from npcpy.ft.sft import run_sft, load_sft_model, predict_sft

X_train = ["translate to french: hello", "translate to french: goodbye"]
y_train = ["bonjour", "au revoir"]

model_path = run_sft(X_train, y_train)

model, tokenizer = load_sft_model(model_path)
response = predict_sft(model, tokenizer, "translate to french: thanks")
```
### Unsupervised Fine-Tuning (USFT)
Adapt models to domain-specific text corpora without labels:
```python
from npcpy.ft.usft import run_usft, load_corpus_from_hf

texts = load_corpus_from_hf("tiny_shakespeare", split="train[:1000]")

model_path = run_usft(
    texts,
    config=USFTConfig(
        output_model_path="models/shakespeare",
        num_train_epochs=3
    )
)
Train on your own text corpus:
pythondomain_texts = [
    "Your domain-specific text 1",
    "Your domain-specific text 2",
] * 100

model_path = run_usft(domain_texts)
```
### Diffusion Fine-tuning
```
from npcpy.ft.diff import train_diffusion, generate_image

image_paths = ["img1.png", "img2.png", "img3.png"]
captions = ["a cat", "a dog", "a bird"]

model_path = train_diffusion(
    image_paths,
    captions,
    config=DiffusionConfig(
        num_epochs=100,
        batch_size=4
    )
)

generated = generate_image(
    model_path,
    prompt="a white square",
    image_size=128
)
Resume training from checkpoint:
pythonmodel_path = train_diffusion(
    image_paths,
    captions,
    config,
    resume_from="models/diffusion/checkpoints/checkpoint-epoch10-step1000.pt"
)
```


### Reinforcement Learning (RL)
Collect agent traces and train with DPO based on reward signals:
```python
from npcpy.ft.rl import collect_traces, run_rl_training
from npcpy.npc_compiler import NPC

tasks = [
    {'prompt': 'Solve 2+2', 'expected': '4'},
    {'prompt': 'Solve 5+3', 'expected': '8'}
]

agents = [
    NPC(name="farlor", primary_directive="Be concise", 
        model="qwen3:0.6b", provider="ollama"),
    NPC(name="tedno", primary_directive="Show your work",
        model="qwen3:0.6b", provider="ollama")
]

def reward_fn(trace):
    if trace['task_metadata']['expected'] in trace['final_output']:
        return 1.0
    return 0.0

adapter_path = run_rl_training(tasks, agents, reward_fn)
```
### Genetic Evolution

Evolve populations of knowledge graphs or model ensembles:
```python
from npcpy.ft.ge import GeneticEvolver, GAConfig

config = GAConfig(
    population_size=20,
    generations=50,
    mutation_rate=0.15
)

evolver = GeneticEvolver(
    fitness_fn=your_fitness_function,
    mutate_fn=your_mutation_function,
    crossover_fn=your_crossover_function,
    initialize_fn=your_init_function,
    config=config
)

best_individual = evolver.run()
```

### Smart Model Ensembler and response router 
Build fast intuitive responses with fallback to reasoning:
```python
from npcpy.ft.model_ensembler import (
    ResponseRouter,
    create_model_genome
)

genome = create_model_genome(['math', 'code', 'factual'])
router = ResponseRouter(fast_threshold=0.8)

result = router.route_query("What is 2+2?", genome)

if result['used_fast_path']:
    print("Fast gut reaction")
elif result['used_ensemble']:
    print("Ensemble voting")
else:
    print("Full reasoning")
```
The intention for this model ensembler system is to mimic human cognition: pattern-matched gut reactions (System 1 of Kahneman) for familiar queries, falling back to deliberate reasoning (System 2 of Kahneman) for novel problems. Genetic algorithms evolve both knowledge structures and model specializations over time.


## NPCArray - NumPy for AI

`npcpy` provides `NPCArray`, a NumPy-like interface for working with populations of models (LLMs, sklearn, PyTorch) at scale. Think of it as vectorized operations over AI models.

### Core Concepts
- Model arrays support vectorized operations
- Operations are lazy until `.collect()` is called (like Spark)
- Same interface works for single models (treated as length-1 arrays)
- Supports ensemble voting, consensus, evolution, and more

### Basic Usage

```python
from npcpy.npc_array import NPCArray

# Create array of LLMs
models = NPCArray.from_llms(
    ['llama3.2', 'gemma3:1b'],
    providers='ollama'
)

print(f"Model array shape: {models.shape}")  # (2,)

# Inference across all models - returns shape (n_models, n_prompts)
result = models.infer("What is 2+2? Just the number.").collect()

print(f"Model 1: {result.data[0, 0]}")
print(f"Model 2: {result.data[1, 0]}")
```

### Lazy Chaining & Ensemble Operations

```python
from npcpy.npc_array import NPCArray

models = NPCArray.from_llms(['llama3.2', 'gemma3:1b', 'mistral:7b'])

# Build lazy computation graph - nothing executed yet
result = (
    models
    .infer("Is Python compiled or interpreted? One word.")
    .map(lambda r: r.strip().lower())  # Clean responses
    .vote(axis=0)  # Majority voting across models
)

# Show the computation plan
result.explain()

# Now execute
answer = result.collect()
print(f"Consensus: {answer.data[0]}")
```

### Parameter Sweeps with Meshgrid

```python
from npcpy.npc_array import NPCArray

# Cartesian product over parameters
configs = NPCArray.meshgrid(
    models=['llama3.2', 'gemma3:1b'],
    temperatures=[0.0, 0.5, 1.0]
)

print(f"Config array shape: {configs.shape}")  # (6,) = 2 models × 3 temps

# Run inference with each config
result = configs.infer("Complete: The quick brown fox").collect()
```

### Matrix Sampling with get_llm_response

The `get_llm_response` function supports `matrix` and `n_samples` parameters for exploration:

```python
from npcpy.llm_funcs import get_llm_response

# Matrix parameter - cartesian product over specified params
result = get_llm_response(
    "Write a creative opening line.",
    matrix={
        'model': ['llama3.2', 'gemma3:1b'],
        'temperature': [0.5, 1.0]
    }
)
print(f"Number of runs: {len(result['runs'])}")  # 4 = 2×2

# n_samples - multiple samples from same config
result = get_llm_response(
    "Pick a random number 1-100.",
    model='llama3.2',
    n_samples=5
)
print(f"Samples: {[r['response'] for r in result['runs']]}")

# Combine both for full exploration
result = get_llm_response(
    "Flip a coin: heads or tails?",
    matrix={'model': ['llama3.2', 'gemma3:1b']},
    n_samples=3  # 2 models × 3 samples = 6 runs
)
```

### sklearn Integration

```python
from npcpy.npc_array import NPCArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# Create sample data
X_train = np.random.randn(100, 4)
y_train = (X_train[:, 0] > 0).astype(int)

# Pre-fit models
rf = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
lr = LogisticRegression().fit(X_train, y_train)

# Create array from fitted models
models = NPCArray.from_sklearn([rf, lr])

# Vectorized prediction
X_test = np.random.randn(20, 4)
predictions = models.predict(X_test).collect()

print(f"RF predictions: {predictions.data[0]}")
print(f"LR predictions: {predictions.data[1]}")
```

### ML Functions with Grid Search

```python
from npcpy.ml_funcs import fit_model, score_model, ensemble_predict

# Grid search via matrix parameter
result = fit_model(
    X_train, y_train,
    model='RandomForestClassifier',
    matrix={
        'n_estimators': [10, 50, 100],
        'max_depth': [3, 5, 10]
    }
)

print(f"Fitted {len(result['models'])} model configurations")

# Ensemble voting with multiple models
predictions = ensemble_predict(X_test, result['models'], method='vote')
```

### Quick Utilities

```python
from npcpy.npc_array import infer_matrix, ensemble_vote

# Quick matrix inference
result = infer_matrix(
    prompts=["Hello", "Goodbye"],
    models=['llama3.2', 'gemma3:1b']
)

# Quick ensemble vote
answer = ensemble_vote(
    "What is the capital of France? One word.",
    models=['llama3.2', 'gemma3:1b']
)
print(f"Voted answer: {answer}")
```

See `examples/npc_array_examples.py` for more comprehensive examples.


## Serving an NPC Team

`npcpy` includes a built-in Flask server that makes it easy to deploy NPC teams for production use. You can serve teams with tools, jinxs, and complex workflows that frontends can interact with via REST APIs.

### Basic Team Server Setup

```python
from npcpy.serve import start_flask_server
from npcpy.npc_compiler import NPC, Team
from npcpy.tools import auto_tools
import requests
import os

# Create NPCs with different specializations
researcher = NPC(
    name='Research Specialist',
    primary_directive='You are a research specialist who finds and analyzes information from various sources.',
    model='claude-3-5-sonnet-latest',
    provider='anthropic'
)

analyst = NPC(
    name='Data Analyst',
    primary_directive='You are a data analyst who processes and interprets research findings.',
    model='gpt-4o',
    provider='openai'
)

coordinator = NPC(
    name='Project Coordinator',
    primary_directive='You coordinate team activities and synthesize results into actionable insights.',
    model='gemini-1.5-pro',
    provider='gemini'
)

# Create team
research_team = Team(
    npcs=[researcher, analyst],
    forenpc=coordinator
)

if __name__ == "__main__":
    # Register team and NPCs directly with the server
    npcs = {npc.name: npc for npc in list(research_team.npcs.values()) + [research_team.forenpc]}
    start_flask_server(
        port=5337,
        cors_origins=["http://localhost:3000", "http://localhost:5173"],  # Allow frontend access
        debug=True,
        teams={'research_team': research_team},
        npcs=npcs
    )
```



## Read the Docs

For more examples of how to use `npcpy` to simplify your LLM workflows  or to create agents or multi-agent systems, read the docs at [npcpy.readthedocs.io](https://npcpy.readthedocs.io/en/latest/)


## Inference Capabilities
- `npcpy` works with local and enterprise LLM providers through its LiteLLM integration, allowing users to run inference from Ollama, LMStudio, OpenAI, Anthropic, Gemini, and Deepseek, making it a versatile tool for both simple commands and sophisticated AI-driven tasks. 



## NPC Studio
There is a graphical user interface that makes use of the NPC Toolkit through the NPC Studio. See the source code for NPC Studio [here](https://github.com/cagostino/npc-studio). Download the executables at [our website](https://enpisi.com/npc-studio).

## NPC Shell

The NPC shell is a suite of executable command-line programs that allow users to easily interact with NPCs and LLMs through a command line shell. 
[Try out the NPC Shell](https://github.com/npc-worldwide/npcsh)


## Mailing List
Interested to stay in the loop and to hear the latest and greatest about `npcpy`, `npcsh` and NPC Studio? Be sure to sign up for the [newsletter](https://forms.gle/n1NzQmwjsV4xv1B2A)!



## Support
If you appreciate the work here, [consider supporting NPC Worldwide with a monthly donation](https://buymeacoffee.com/npcworldwide), [buying NPC-WW themed merch](https://enpisi.com/shop), or hiring us to help you explore how to use `npcpy` and AI tools to help your business or research team, please reach out to info@npcworldwi.de .





## Enabling Innovation and Research
- `npcpy` is a framework that speeds up and simplifies the development of NLP-based or Agent-based applications and provides developers and researchers with methods to explore and test across dozens of models, providers, and personas as well as other model-level hyperparameters (e.g. `temperature`, `top_k`, etc.), incorporating an array of data sources and common tools.
- The `npcpy` agent data layer makes it easy to set up teams and serve them so you can focus more on the agent personas and less on the nitty gritty of inference.
- `npcpy` provides pioneering methods in the construction and updating of knowledge graphs as well as in the development and testing of novel mixture of agent scenarios.
- In `npcpy`, all agentic capabilities are developed and tested using small local models (like `llama3.2`, `gemma3`) to ensure it can function reliably at the edge of computing.

### Papers
- Paper on the limitations of LLMs and on the quantum-like nature of natural language interpretation : [arxiv preprint](https://arxiv.org/abs/2506.10077), accepted for publication at [Quantum AI and NLP 2025](qnlp.ai)
- Paper that considers the effects that might accompany simulating hormonal cycles for AI : [arxiv preprint](https://arxiv.org/abs/2508.11829)

Has your research benefited from npcpy? Let us know and we'd be happy to feature you here!

## NPCs

Check out [lavanzaro](https://lavanzaro.com) to discuss the great things of life with an `npcpy` powered chatbot

## Installation
`npcpy` is available on PyPI and can be installed using pip. Before installing, make sure you have the necessary dependencies installed on your system. Below are the instructions for installing such dependencies on Linux, Mac, and Windows. If you find any other dependencies that are needed, please let us know so we can update the installation instructions to be more accommodating.

### Linux install
<details>  <summary> Toggle </summary>
  
```bash

# these are for audio primarily, skip if you dont need tts
sudo apt-get install espeak
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install alsa-base alsa-utils
sudo apt-get install libcairo2-dev
sudo apt-get install libgirepository1.0-dev
sudo apt-get install ffmpeg

# for triggers
sudo apt install inotify-tools


#And if you don't have ollama installed, use this:
curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcpy
# if you want to install with the API libraries
pip install 'npcpy[lite]'
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install 'npcpy[local]'
# if you want to use tts/stt
pip install 'npcpy[yap]'
# if you want everything:
pip install 'npcpy[all]'

```

</details>


### Mac install

<details>  <summary> Toggle </summary>

```bash
#mainly for audio
brew install portaudio
brew install ffmpeg
brew install pygobject3

# for triggers
brew install inotify-tools


brew install ollama
brew services start ollama
ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcpy
# if you want to install with the API libraries
pip install npcpy[lite]
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install npcpy[local]
# if you want to use tts/stt
pip install npcpy[yap]

# if you want everything:
pip install npcpy[all]
```
</details>

### Windows Install

<details>  <summary> Toggle </summary>
Download and install ollama exe.

Then, in a powershell. Download and install ffmpeg.

```powershell
ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcpy
# if you want to install with the API libraries
pip install npcpy[lite]
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install npcpy[local]
# if you want to use tts/stt
pip install npcpy[yap]

# if you want everything:
pip install npcpy[all]
```

</details>

### Fedora Install (under construction)

<details>  <summary> Toggle </summary>
  
```bash
python3-dev #(fixes hnswlib issues with chroma db)
xhost +  (pyautogui)
python-tkinter (pyautogui)
```

</details>


We support inference via all providers supported by litellm. For openai-compatible providers that are not explicitly named in litellm, use simply `openai-like` as the provider. The default provider must be one of `['openai','anthropic','ollama', 'gemini', 'deepseek', 'openai-like']` and the model must be one available from those providers.

To use tools that require API keys, create an `.env` file in the folder where you are working or place relevant API keys as env variables in your ~/.npcshrc. If you already have these API keys set in a ~/.bashrc or a ~/.zshrc or similar files, you need not additionally add them to ~/.npcshrc or to an `.env` file. Here is an example of what an `.env` file might look like:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export DEEPSEEK_API_KEY='your_deepseek_key'
export GEMINI_API_KEY='your_gemini_key'
export PERPLEXITY_API_KEY='your_perplexity_key'
```


 Individual npcs can also be set to use different models and providers by setting the `model` and `provider` keys in the npc files.


For cases where you wish to set up a team of NPCs, jinxs, and assembly lines, add a `npc_team` directory to your project and then initialize an NPC Team.
```bash
./npc_team/            # Project-specific NPCs
├── jinxs/             # Project jinxs #example jinx next
│   └── example.jinx
└── assembly_lines/    # Project workflows
    └── example.pipe
└── models/    # Project workflows
    └── example.model
└── example1.npc        # Example NPC
└── example2.npc        # Example NPC
└── team.ctx            # Example ctx


```


## Contributing
Contributions are welcome! Please submit issues and pull requests on the GitHub repository.


## License
This project is licensed under the MIT License.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=cagostino/npcpy&type=Date)](https://star-history.com/#cagostino/npcpy&Date)
