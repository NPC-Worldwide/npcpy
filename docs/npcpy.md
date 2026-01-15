# NPCPY


## Python Examples
Integrate `npcpy` into your Python projects for additional flexibility. Below are a few examples of how to use the library programmatically.

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
output= content
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


# Define a jinx for literary research
literary_research_jinx = Jinx(jinx_data={
    "jinx_name": "literary_research",
    "description": "Research a literary topic, analyze files, and summarize findings",
    "inputs": ["topic"],
    "steps": [
        {
            "name": "gather_info",
            "engine": "natural",
            "code": """
                Research the topic: {{ topic }}.
                Summarize the main themes and historical context.
            """
        },
        {
            "name": "final_summary",
            "engine": "natural",
            "code": """
                Based on the research in. {{gather_info}}, write a concise, creative summary.
            """
        }
    ]
})

ggm = NPC(
    name='Gabriel Garcia Marquez',
    primary_directive='You are Gabriel Garcia Marquez, master of magical realism. Research, analyze, and write with poetic flair.',
    model='gemma3:4b',
    provider='ollama',
)

isabel = NPC(
    name='Isabel Allende',
    primary_directive='You are Isabel Allende, weaving stories with emotion and history. Analyze texts and provide insight.',
    model='llama3.2',
    provider='ollama',

)

borges = NPC(
    name='Jorge Luis Borges',
    primary_directive='You are Borges, philosopher of labyrinths and libraries. Synthesize findings and create literary puzzles.',
    model='qwen3:latest',
    provider='ollama',
)

# Set up a team with a forenpc that orchestrates the other npcs
lit_team = Team(npcs=[ggm, isabel], forenpc=borges, jinxs={'literary_research': literary_research_jinx, 'file_reader': file_reader_jinx},
)

# Example: Orchestrate a jinx workflow
result = lit_team.orchestrate(
    "Research the topic of magical realism, read ./test_data/magical_realism.txt and summarize the findings"
)
print(result['debrief']['summary'])

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
response = get_llm_response("Who was the celtic Messenger god?", model='qwen3:4b', provider='ollama')
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
response = get_llm_response("What is the sentiment of the american people towards the repeal of Roe v Wade? Return a json object with `sentiment` as the key and a float value from -1 to 1 as the value", model='deepseek-chat', provider='deepseek', format='json')

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

### Multimedia Production Team with Tools

Here's a complete example of a multimedia team with image generation, editing, web search, screenshot capabilities, desktop automation, data analysis, and workflow planning. The key is to properly register both the team and individual NPCs with the server so they can be accessed via the API:

```python
from npcpy.serve import start_flask_server
from npcpy.npc_compiler import NPC, Team
from npcpy.tools import auto_tools
from npcpy.gen.image_gen import generate_image, edit_image
from npcpy.data.web import search_web
from npcpy.data.image import capture_screenshot
from npcpy.data.load import load_csv, load_pdf, load_json, load_excel
from npcpy.work.desktop import perform_action
from npcpy.work.plan import execute_plan_command
from npcpy.work.trigger import execute_trigger_command
import pandas as pd

# Create specialized NPCs with comprehensive tool sets
image_generator = NPC(
    name='Image Creator',
    primary_directive='You are a creative image generator who creates stunning visuals based on descriptions and requirements.',
    model='gpt-4o',
    provider='openai',
    tools=[generate_image]
)

image_editor = NPC(
    name='Image Editor',
    primary_directive='You are a skilled image editor who enhances and modifies images to meet specific requirements.',
    model='claude-3-5-sonnet-latest', 
    provider='anthropic',
    tools=[edit_image]
)

web_researcher = NPC(
    name='Web Researcher',
    primary_directive='You are a thorough web researcher who finds relevant information and visual references online.',
    model='deepseek-chat',
    provider='deepseek',
    tools=[search_web]
)

screenshot_specialist = NPC(
    name='Screenshot Specialist',
    primary_directive='You are a screenshot specialist who captures screen content and visual elements as needed.',
    model='llama3.2',
    provider='ollama',
    tools=[capture_screenshot]
)

desktop_automator = NPC(
    name='Desktop Automation Specialist',
    primary_directive='You are a desktop automation specialist who can control the computer interface, click, type, and perform various desktop actions.',
    model='qwen3:latest',
    provider='ollama',
    tools=[perform_action]
)

data_analyst = NPC(
    name='Data Analyst',
    primary_directive='You are a data analyst who can load, process, and analyze various data formats including CSV, Excel, PDF, and JSON files.',
    model='gemini-1.5-pro',
    provider='gemini',
    tools=[load_csv, load_pdf, load_json, load_excel]
)

workflow_planner = NPC(
    name='Workflow Planner',
    primary_directive='You are a workflow automation specialist who can create scheduled tasks and automated triggers for various system operations.',
    model='claude-3-5-sonnet-latest',
    provider='anthropic',
    tools=[execute_plan_command, execute_trigger_command]
)

multimedia_coordinator = NPC(
    name='Multimedia Director',
    primary_directive='You are a multimedia director who coordinates image creation, editing, research, data analysis, desktop automation, and workflow planning.',
    model='gemini-2.0-flash',
    provider='gemini'
)

# Create comprehensive multimedia production team
multimedia_team = Team(
    npcs=[
        image_generator,
        image_editor,
        web_researcher, 
        screenshot_specialist,
        desktop_automator,
        data_analyst,
        workflow_planner
    ],
    forenpc=multimedia_coordinator
)

# Start server for multimedia team
if __name__ == "__main__":
    # Register team and NPCs directly with the server
    npcs = {npc.name: npc for npc in list(multimedia_team.npcs.values()) + [multimedia_team.forenpc]}
    start_flask_server(
        port=5337,
        cors_origins=["*"],  # Allow all origins for development
        debug=True,
        teams={'multimedia_team': multimedia_team},
        npcs=npcs
    )

```

**Important:** The `teams` and `npcs` parameters in `start_flask_server()` register your teams and NPCs with the server so they can be accessed via the API endpoints. When you make requests with `"team": "multimedia_team"`, the server will use the registered team object with all its tools and capabilities.

**NPC Loading Priority:** When specifying both `"team"` and `"npc"` in API requests, the server will:
1. First look for the NPC within the specified registered team
2. Then check globally registered NPCs 
3. Finally fall back to loading NPCs from database/files

This means you can access individual NPCs within a team (like `"Image Creator"` from `"multimedia_team"`) and they will have all of their registered tools and capabilities. The registered NPCs include all their tools and can be accessed individually via the API, or as part of a team orchestration through the team's forenpc coordinator.

### Frontend Integration Examples

Once your server is running, frontends can interact with your NPC teams via REST API:

#### curl Examples

```bash
# Execute a comprehensive multimedia team task
curl -X POST http://localhost:5337/api/execute \
  -H "Content-Type: application/json" \
  -d '{
    "commandstr": "Search for the latest tech startup trends and take a screenshot of the current desktop",
    "team": "multimedia_team",
    "conversationId": "curl_session_123", 
    "provider": "ollama", 
    "model": "llama3.2"
  }'

# Call a specific NPC within the multimedia team
curl -X POST http://localhost:5337/api/execute \
  -H "Content-Type: application/json" \
  -d '{
    "commandstr": "Search for fintech startup market analysis reports",
    "team": "multimedia_team",
    "npc": "Web Researcher",
    "conversationId": "search_session", 
    "model": "llama3.2",
    "provider": "ollama" 
  }'

# Stream a team response for real-time updates
curl -X POST http://localhost:5337/api/stream \
  -H "Content-Type: application/json" \
  -d '{
    "commandstr": "Search for design inspiration websites and load any CSV files in the current directory",
    "team": "multimedia_team",
    "conversationId": "streaming_session", 
    "model": "llama3.2",
    "provider": "ollama" 
  }' 


# Get available models
curl -X GET http://localhost:5337/api/models

# List available NPCs in the team
curl -X GET http://localhost:5337/api/npc_team_global

# List available jinxs
curl -X GET http://localhost:5337/api/jinxs/global

# Execute a specific jinx with inputs
curl -X POST http://localhost:5337/api/execute \
  -H "Content-Type: application/json" \
  -d '{
    "commandstr": "Execute the content_creation_pipeline jinx",
    "jinx_name": "content_creation_pipeline",
    "jinx_inputs": {
      "topic": "AI in Healthcare",
      "target_audience": "medical professionals", 
      "content_type": "infographic"
    },
    "conversationId": "jinx_session"
  }'

# Get conversation history
curl -X GET "http://localhost:5337/api/messages?conversationId=curl_session_123&limit=10"

# Health check endpoint
curl -X GET http://localhost:5337/api/health
```

#### JavaScript/React Example

```javascript
// Send request to comprehensive multimedia team
const requestMultimediaWork = async (task) => {
  const response = await fetch('http://localhost:5337/api/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      commandstr: task,
      team: 'multimedia_team',
      conversationId: 'session_123'
    })
  });
  
  const result = await response.json();
  return result;
};

// Example usage - quick web research workflow
const result = await requestMultimediaWork(
  "Search for the latest AI trends in 2024, then take a screenshot of the current desktop"
);

// Data loading and analysis example
const dataResult = await requestMultimediaWork(
  "Load any CSV files from the current directory and provide a summary of the data structure"
);

// Web search and file operations
const searchResult = await requestMultimediaWork(
  "Search for 'best practices in web development' and save the results summary to a text file"
);
const analysisResult = await requestMultimediaWork(
  "Load the market_research.pdf, extract key insights, search for related trends online, and generate an infographic summarizing the findings"
);
```

#### Python Client Example

```python
import requests

def use_multimedia_team(task, base_url="http://localhost:5337"):
    """Send a task to the comprehensive multimedia team server."""
    response = requests.post(f"{base_url}/api/execute", json={
        "commandstr": task,
        "team": "multimedia_team", 
        "conversationId": "python_client_session"
    })
    return response.json()

# Quick web research workflow example
result = use_multimedia_team(
    "Search for the latest developments in AI technology and provide a summary of key findings"
)
print(result['debrief']['summary'])

# Data loading and analysis example
data_result = use_multimedia_team(
    "Load any CSV files in the current directory and provide an analysis of the data structure and contents"
)
print(data_result['debrief']['summary'])

# Web search and file operations
search_result = use_multimedia_team(
    "Search for information about machine learning best practices and save the key points to a text file"
)
print(search_result['debrief']['summary'])

# Workflow automation setup
automation_result = use_multimedia_team(
    "Set up a daily task to check for new CSV files in Downloads folder, automatically process them for analysis, generate summary reports, and trigger email notifications"
)
print(automation_result['debrief']['summary'])
```

### Team with Jinxs for Complex Workflows

```python
from npcpy.npc_compiler import NPC, Team, Jinx

# Create a comprehensive multimedia analysis jinx
multimedia_analysis_jinx = Jinx(jinx_data={
    "jinx_name": "multimedia_content_pipeline",
    "description": "Complete multimedia content creation and analysis workflow",
    "inputs": ["topic", "data_sources", "output_formats", "automation_schedule"],
    "steps": [
        {
            "name": "research_and_data_gathering",
            "engine": "natural",
            "code": """
            Research the topic "{{ topic }}" comprehensively.
            Load and analyze data from the specified sources: {{ data_sources }}.
            Search for current trends, competitors, and market insights online.
            Capture screenshots of relevant examples and references.
            
            Provide comprehensive research including:
            - Current market trends and developments
            - Competitor analysis with visual examples
            - Data insights from loaded files
            - Visual references and inspiration
            """
        },
        {
            "name": "data_processing_and_analysis",
            "engine": "python",
            "code": """
            import pandas as pd
            import numpy as np
            from datetime import datetime
            
            research = context.get('llm_response', '')
            data_sources = '{{ data_sources }}'.split(',')
            
            # Process each data source
            processed_data = {}
            for source in data_sources:
                source = source.strip()
                if source.endswith('.csv'):
                    try:
                        df = pd.read_csv(source)
                        processed_data[source] = {
                            'shape': df.shape,
                            'columns': list(df.columns),
                            'summary': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else 'No numeric data',
                            'sample': df.head().to_dict()
                        }
                    except Exception as e:
                        processed_data[source] = f'Error loading: {str(e)}'
                elif source.endswith('.xlsx'):
                    try:
                        df = pd.read_excel(source)
                        processed_data[source] = {
                            'shape': df.shape,
                            'columns': list(df.columns),
                            'summary': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else 'No numeric data'
                        }
                    except Exception as e:
                        processed_data[source] = f'Error loading: {str(e)}'
            
            context['processed_data'] = processed_data
            context['analysis_timestamp'] = datetime.now().isoformat()
            
            output = f"Data processing complete. Analyzed {len(data_sources)} sources at {context['analysis_timestamp']}"
            """
        },
        {
            "name": "content_creation",
            "engine": "natural",
            "code": """
            Based on the research and data analysis results:
            
            Research insights: {{ llm_response }}
            Processed data: {{ processed_data }}
            Analysis timestamp: {{ analysis_timestamp }}
            
            Create multimedia content for "{{ topic }}" in the following formats: {{ output_formats }}.
            
            Generate appropriate visuals including:
            - Data visualization charts and infographics
            - Hero images and promotional graphics  
            - Social media content variations
            - Presentation slides and diagrams
            
            Edit and enhance images as needed for professional quality.
            Ensure all content aligns with the research insights and data findings.
            """
        },
        {
            "name": "automation_setup",
            "engine": "natural", 
            "code": """
            Set up automated workflows based on the schedule: {{ automation_schedule }}.
            
            Create the following automation:
            1. Scheduled tasks for regular data updates and analysis
            2. Triggers for monitoring new data files
            3. Automated report generation and distribution
            4. Content refresh and social media posting schedules
            
            Configure desktop automation for:
            - File organization and processing
            - Application launching and data loading
            - Screenshot capture for monitoring
            - Notification and alert systems
            """
        }
    ]
})

# Create comprehensive content team with expanded capabilities
multimedia_creator = NPC(
    name='Multimedia Creator',
    primary_directive='You are a multimedia creator who develops engaging content across multiple formats and platforms.',
    model='gpt-4o',
    provider='openai',
    jinxs=[multimedia_analysis_jinx],
    tools=[generate_image, edit_image, capture_screenshot]
)

data_processor = NPC(
    name='Data Processing Specialist', 
    primary_directive='You are a data processing specialist who loads, analyzes, and transforms data from various sources.',
    model='claude-3-5-sonnet-latest',
    provider='anthropic',
    tools=[load_csv, load_excel, load_pdf, load_json]
)

automation_engineer = NPC(
    name='Automation Engineer',
    primary_directive='You are an automation engineer who creates workflows, schedules, and desktop automation solutions.',
    model='gemini-1.5-pro',
    provider='gemini',
    tools=[execute_plan_command, execute_trigger_command, perform_action]
)

web_intelligence = NPC(
    name='Web Intelligence Specialist',
    primary_directive='You are a web intelligence specialist who gathers market insights and competitive analysis.',
    model='deepseek-chat',
    provider='deepseek',
    tools=[search_web]
)

content_director = NPC(
    name='Content Director',
    primary_directive='You are a content director who oversees comprehensive multimedia production workflows and automation systems.',
    model='gemini-2.0-flash',
    provider='gemini'
)

comprehensive_team = Team(
    npcs=[multimedia_creator, data_processor, automation_engineer, web_intelligence],
    forenpc=content_director
)

# Execute comprehensive multimedia pipeline
result = comprehensive_team.orchestrate(
    "Execute the multimedia_content_pipeline jinx for topic='Sustainable Technology Trends 2025', data_sources='market_data.csv,research_report.pdf', output_formats='infographics,social_media,presentation', automation_schedule='daily_analysis,weekly_reports'"
)
print(result['debrief']['summary'])
```

## Server API Endpoints
The server provides REST endpoints for:
- `/api/execute` - Execute team commands
- `/api/stream` - Stream team responses  
- `/api/models` - Get available models
- `/api/npc_team_global` - List available NPCs
- `/api/jinxs/global` - List available jinxs

This setup allows you to quickly deploy sophisticated NPC teams with multimedia capabilities, complex workflows, and tool integrations that frontends can consume via standard REST APIs.




## Tool Calling Examples

`npcpy` supports tool calling both with and without NPCs, allowing you to extend LLM capabilities with custom functions.

### Tool Calling without NPCs

```python
from npcpy.llm_funcs import get_llm_response
from npcpy.tools import auto_tools
import os

def create_file(filename: str, content: str) -> str:
    """Create a new file with content."""
    with open(filename, 'w') as f:
        f.write(content)
    return f"Created {filename}"

def count_files(directory: str = ".") -> int:
    """Count files in a directory."""
    return len([f for f in os.listdir(directory) if os.path.isfile(f)])

def get_file_size(filename: str) -> str:
    """Get the size of a file."""
    size = os.path.getsize(filename)
    return f"{filename} is {size} bytes"

# Auto-generate tools
tools_schema, tool_map = auto_tools([create_file, count_files, get_file_size])

# Let the LLM use multiple tools naturally
response = get_llm_response(
    "Create a file called 'hello.txt' with the content 'Hello World!', then tell me how many files are in the current directory and what size the new file is",
    model='deepseek-reasoner',
    provider='deepseek',
    tools=tools_schema,
    tool_map=tool_map
)

# Process the multi-tool workflow results
print("Multi-tool workflow executed:")
for i, (call, result) in enumerate(zip(response.get('tool_calls', []), response.get('tool_results', []))):
    func_name = call['function']['name']
    args = call['function']['arguments']
    print(f"{i+1}. {func_name}({args}) → {result}")

# Example of building a summary from tool results
if response.get('tool_results'):
    file_created = any('Created' in str(result) for result in response['tool_results'])
    file_count = next((result for result in response['tool_results'] if isinstance(result, int)), None)
    file_size = next((result for result in response['tool_results'] if 'bytes' in str(result)), None)
    
    summary = f"Summary: File created: {file_created}, Directory has {file_count} files, {file_size}"
    print(f"\n{summary}")
```

## Understanding Tool Calling Response Structure

When using tools with `npcpy`, the response structure differs from regular LLM responses. Instead of a synthesized text response, you get structured data about the tools that were called and their results. This allows you to decide what actions to proceed with following the tool call.

### Response Object Structure

```python
response = {
    'response': None,                    # Always None when tools are used
    'tool_calls': [...],                 # List of tools the LLM decided to call
    'tool_results': [...],               # Results from executing those tools
    'messages': [...],                   # Full conversation history with tool interactions
    'usage': {...},                      # Token usage information
    'model': 'gpt-4o-mini',             # Model used
    'provider': 'openai'                 # Provider used
}
```



## Jinx Examples

Jinja execution templates---Jinxs---are powerful workflow templates that combine natural language processing with Python code execution through sequential processing of steps. They're defined in YAML files and can be used by NPCs or called directly. Jinxs can reference other Jinxs through Jinja references, allowing for modular and reusable workflows 
that can be easily shared, adapted, and extended.

### Creating a Simple Jinx

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
      
      # Load the CSV file
      df = pd.read_csv('{{ file_path }}')
      print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
      
      # Store in context for next steps
      context['dataframe'] = df
      context['row_count'] = len(df)
      context['column_count'] = len(df.columns)
      
  - name: "analyze_data"
    engine: "python" 
    code: |
      df = context['dataframe']
      analysis_type = '{{ analysis_type }}'.lower()
      
      if analysis_type == 'basic':
          # Basic statistics
          stats = df.describe()
          context['statistics'] = stats.to_dict()
          output = f"Basic statistics computed for {len(df.columns)} columns"
      elif analysis_type == 'correlation':
          # Correlation analysis
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
      
      Please generate a comprehensive summary report of the key findings and insights.
```

### Using Jinx with NPCs

```python
from npcpy.npc_compiler import NPC, Jinx

# Create NPC with jinx
data_scientist = NPC(
    name='Data Scientist',
    primary_directive='You are an expert data scientist specializing in data analysis and insights.',
    jinxs=['data_analyzer'],  # Reference the jinx file
    model='llama3.2:13b',
    provider='ollama'
)

# Execute the jinx
result = data_scientist.execute_jinx(
    'data_analyzer',
    {
        'file_path': './sales_data.csv',
        'analysis_type': 'basic'
    }
)

print(result['output'])
```

### Complex Jinx with Multiple Steps

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
      Please research the topic: {{ research_topic }}
      
      Provide comprehensive information including:
      1. Key concepts and definitions
      2. Current trends and developments
      3. Major challenges or controversies
      4. Future outlook
      
      Focus on recent, credible sources and provide specific examples.
      
  - name: "analyze_findings"
    engine: "python"
    code: |
      # Extract key information from the research
      research_text = context.get('llm_response', '')
      
      # Simple analysis - count key terms
      import re
      from collections import Counter
      
      # Extract sentences and key phrases
      sentences = re.split(r'[.!?]', research_text)
      context['sentence_count'] = len([s for s in sentences if len(s.strip()) > 10])
      
      # Find common important terms (simple approach)
      words = re.findall(r'\b[A-Z][a-z]+\b', research_text)
      common_terms = Counter(words).most_common(10)
      context['key_terms'] = dict(common_terms)
      
      output = f"Analysis complete: {context['sentence_count']} sentences, top terms: {list(context['key_terms'].keys())[:5]}"
      
  - name: "format_report"
    engine: "natural" 
    code: |
      Based on the research findings about {{ research_topic }}, create a well-structured report in {{ output_format }} format.
      
      Research Summary:
      {{ llm_response }}
      
      Key Statistics:
      - Number of key points covered: {{ sentence_count }}
      - Most mentioned terms: {{ key_terms }}
      
      Please format this as a professional {{ output_format }} with:
      1. Executive Summary
      2. Main Findings  
      3. Analysis and Insights
      4. Recommendations
      5. Conclusion
      
      Ensure the content is well-organized and actionable.
```

### Using Jinxs Directly

```python
from npcpy.npc_compiler import Jinx, NPC

# Load and execute jinx directly
research_jinx = Jinx(jinx_path='./research_pipeline.jinx')

# Create a simple NPC for execution
npc = NPC(
    name='Research Assistant',
    primary_directive='You are a research assistant specialized in analyzing and reporting on various topics.',
    model='gemini-1.5-pro',
    provider='gemini'
)

# Execute the jinx
result = research_jinx.execute(
    input_values={
        'research_topic': 'artificial intelligence in healthcare',
        'output_format': 'markdown'
    },
    jinxs_dict={'research_pipeline': research_jinx},
    npc=npc
)

print(result['output'])
```

### Team-based Jinx Usage

```python
from npcpy.npc_compiler import NPC, Team

# Create specialized NPCs for different tasks
researcher = NPC(
    name='Researcher',
    primary_directive='You are a thorough researcher who gathers comprehensive information.',
    jinxs=['research_pipeline'],
    model='gemini-2.0-flash',
    provider='gemini'
)

analyst = NPC(
    name='Data Analyst', 
    primary_directive='You are a data analyst who excels at finding patterns and insights.',
    jinxs=['data_analyzer'],
    model='claude-3-5-sonnet-latest',
    provider='anthropic'
)

writer = NPC(
    name='Technical Writer',
    primary_directive='You are a technical writer who creates clear, well-structured documents.',
    model='llama3.2',
    provider='ollama'
)

# Create team with forenpc (coordinator)
research_team = Team(
    npcs=[researcher, analyst],
    forenpc=writer
)

# Orchestrate complex workflow
result = research_team.orchestrate(
    "Research the impact of AI in education, analyze any available data, and create a comprehensive report"
)

print(result)
```
