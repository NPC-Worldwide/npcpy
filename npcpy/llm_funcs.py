from jinja2 import Environment, FileSystemLoader, Undefined
import json
import PIL
import random 
import subprocess
from typing import List, Dict, Any, Optional, Union
from npcpy.npc_sysenv import (
    print_and_process_stream_with_markdown,
    render_markdown,
    lookup_provider,
    request_user_input, 
    get_system_message
)
from npcpy.gen.response import get_litellm_response
from npcpy.gen.image_gen import generate_image
from npcpy.gen.video_gen import generate_video_diffusers, generate_video_veo3

from datetime import datetime 

def gen_image(
    prompt: str,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    height: int = 1024,
    width: int = 1024,
    n_images: int=1, 
    input_images: List[Union[str, bytes, PIL.Image.Image]] = None,
    save = False, 
    filename = '',
):
    """This function generates an image using the specified provider and model.
    Args:
        prompt (str): The prompt for generating the image.
    Keyword Args:
        model (str): The model to use for generating the image.
        provider (str): The provider to use for generating the image.
        filename (str): The filename to save the image to.
        npc (Any): The NPC object.
    Returns:
        str: The filename of the saved image.
    """
    if model is not None and provider is not None:
        pass
    elif model is not None and provider is None:
        provider = lookup_provider(model)
    elif npc is not None:
        if npc.provider is not None:
            provider = npc.provider
        if npc.model is not None:
            model = npc.model
        if npc.api_url is not None:
            api_url = npc.api_url

    images = generate_image(
        prompt=prompt,
        model=model,
        provider=provider,
        height=height,
        width=width, 
        attachments=input_images,
        n_images=n_images, 
        
    )
    if save:
        if len(filename) == 0 :
            todays_date = datetime.now().strftime("%Y-%m-%d")
            filename = 'vixynt_gen'
        for i, image in enumerate(images):
            
            image.save(filename+'_'+str(i)+'.png')
    return images


def gen_video(
    prompt,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    device: str = "cpu",
    output_path="",
    num_inference_steps=10,
    num_frames=25,
    height=256,
    width=256,
    negative_prompt="",
    messages: list = None,
):
    """
    Function Description:
        This function generates a video using either Diffusers or Veo 3 via Gemini API.
    Args:
        prompt (str): The prompt for generating the video.
    Keyword Args:
        model (str): The model to use for generating the video.
        provider (str): The provider to use for generating the video (gemini for Veo 3).
        device (str): The device to run the model on ('cpu' or 'cuda').
        negative_prompt (str): What to avoid in the video (Veo 3 only).
    Returns:
        dict: Response with output path and messages.
    """
    
    if provider == "gemini":
        
        try:
            output_path = generate_video_veo3(
                prompt=prompt,
                negative_prompt=negative_prompt,
                output_path=output_path,
            )
            return {
                "output": f"High-fidelity video with synchronized audio generated at {output_path}",
                "messages": messages
            }
        except Exception as e:
            print(f"Veo 3 generation failed: {e}")
            print("Falling back to diffusers...")
            provider = "diffusers"
    
    if provider == "diffusers" or provider is None:
      
        output_path = generate_video_diffusers(
            prompt,
            model,
            npc=npc,
            device=device,
            output_path=output_path,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            height=height,
            width=width,
        )
        return {
            "output": f"Video generated at {output_path}",
            "messages": messages
        }
    
    return {
        "output": f"Unsupported provider: {provider}",
        "messages": messages
    }


def get_llm_response(
    prompt: str,
    model: str=None,
    provider: str = None,
    images: List[str] = None,
    npc: Any = None,
    team: Any = None,
    messages: List[Dict[str, str]] = None,
    api_url: str = None,
    api_key: str = None,
    context=None,    
    stream: bool = False,
    attachments: List[str] = None,
    include_usage: bool = False,
    **kwargs,
):
    """This function generates a response using the specified provider and model.
    Args:
        prompt (str): The prompt for generating the response.
    Keyword Args:
        provider (str): The provider to use for generating the response.
        model (str): The model to use for generating the response.
        images (List[Dict[str, str]]): The list of images.
        npc (Any): The NPC object.
        messages (List[Dict[str, str]]): The list of messages.
        api_url (str): The URL of the API endpoint.
        attachments (List[str]): List of file paths to include as attachments
    Returns:
        Any: The response generated by the specified provider and model.
    """
    if model is not None and provider is not None:
        pass
    elif provider is None and model is not None:
        provider = lookup_provider(model)
    elif npc is not None:
        if npc.provider is not None:
            provider = npc.provider
        if npc.model is not None:
            model = npc.model
        if npc.api_url is not None:
            api_url = npc.api_url
    elif team is not None:
        if team.model is not None:
            model = team.model
        if team.provider is not None:
            provider = team.provider
        if team.api_url is not None:
            api_url = team.api_url
                
    else:
        provider = "ollama"
        if images is not None or attachments is not None:
            model = "llava:7b"
        else:
            model = "llama3.2"
            
    if npc is not None:
        
        system_message = get_system_message(npc, team) 
    else: 
        system_message = "You are a helpful assistant."
   

    if context is not None:
        context_str = f'User Provided Context: {context}'
    else:
        context_str = ''

    if messages is None or len(messages) == 0:
        messages = [{"role": "system", "content": system_message}]
        if prompt:
            messages.append({"role": "user", "content": prompt+context_str})
    elif prompt and messages[-1]["role"] == "user":

        if isinstance(messages[-1]["content"], str):
            messages[-1]["content"] += "\n" + prompt+context_str
    elif prompt:
        messages.append({"role": "user", 
                         "content": prompt + context_str})    
    #import pdb 
    #pdb.set_trace()
    response = get_litellm_response(
        prompt + context_str,
        messages=messages,
        model=model,
        provider=provider,
        api_url=api_url,
        api_key=api_key,
        images=images,
        attachments=attachments,
        stream=stream,
        include_usage=include_usage,
        **kwargs,
    )
    return response




def execute_llm_command(
    command: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_url: str = None,
    api_key: str = None,
    npc: Optional[Any] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    stream=False,
    context=None,
) -> str:
    """This function executes an LLM command.
    Args:
        command (str): The command to execute.

    Keyword Args:
        model (Optional[str]): The model to use for executing the command.
        provider (Optional[str]): The provider to use for executing the command.
        npc (Optional[Any]): The NPC object.
        messages (Optional[List[Dict[str, str]]): The list of messages.
    Returns:
        str: The result of the LLM command.
    """
    if messages is None:
        messages = []
    max_attempts = 5
    attempt = 0
    subcommands = []

   
    context = ""
    while attempt < max_attempts:
        prompt = f"""
        A user submitted this query: {command}.
        You need to generate a bash command that will accomplish the user's intent.
        Respond ONLY with the bash command that should be executed. 
        Do not include markdown formatting
        """
        response = get_llm_response(
            prompt,
            model=model,
            provider=provider,
            api_url=api_url,
            api_key=api_key,
            messages=messages,
            npc=npc,
            context=context,
        )

        bash_command = response.get("response", {})
 
        print(f"LLM suggests the following bash command: {bash_command}")
        subcommands.append(bash_command)

        try:
            print(f"Running command: {bash_command}")
            result = subprocess.run(
                bash_command, shell=True, text=True, capture_output=True, check=True
            )
            print(f"Command executed with output: {result.stdout}")

            prompt = f"""
                Here was the output of the result for the {command} inquiry
                which ran this bash command {bash_command}:

                {result.stdout}

                Provide a simple response to the user that explains to them
                what you did and how it accomplishes what they asked for.
                """

            messages.append({"role": "user", "content": prompt})
           
            response = get_llm_response(
                prompt,
                model=model,
                provider=provider,
                api_url=api_url,
                api_key=api_key,
                npc=npc,
                messages=messages,
                context=context,
                stream =stream
            )

            return response
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error:")
            print(e.stderr)

            error_prompt = f"""
            The command '{bash_command}' failed with the following error:
            {e.stderr}
            Please suggest a fix or an alternative command.
            Respond with a JSON object containing the key "bash_command" with the suggested command.
            Do not include any additional markdown formatting.

            """

            fix_suggestion = get_llm_response(
                error_prompt,
                model=model,
                provider=provider,
                npc=npc,
                api_url=api_url,
                api_key=api_key,
                format="json",
                messages=messages,
                context=context,
            )

            fix_suggestion_response = fix_suggestion.get("response", {})

            try:
                if isinstance(fix_suggestion_response, str):
                    fix_suggestion_response = json.loads(fix_suggestion_response)

                if (
                    isinstance(fix_suggestion_response, dict)
                    and "bash_command" in fix_suggestion_response
                ):
                    print(
                        f"LLM suggests fix: {fix_suggestion_response['bash_command']}"
                    )
                    command = fix_suggestion_response["bash_command"]
                else:
                    raise ValueError(
                        "Invalid response format from LLM for fix suggestion"
                    )
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing LLM fix suggestion: {e}")

        attempt += 1

    return {
        "messages": messages,
        "output": "Max attempts reached. Unable to execute the command successfully.",
    }

def handle_jinx_call(
    command: str,
    jinx_name: str,
    model: str = None,
    provider: str = None,
    messages: List[Dict[str, str]] = None,
    npc: Any = None,
    team: Any = None,
    stream=False,
    n_attempts=3,
    attempt=0,
    context=None,
    **kwargs
) -> Union[str, Dict[str, Any]]:
    """This function handles a jinx call.
    Args:
        command (str): The command.
        jinx_name (str): The jinx name.
    Keyword Args:
        model (str): The model to use for handling the jinx call.
        provider (str): The provider to use for handling the jinx call.
        messages (List[Dict[str, str]]): The list of messages.
        npc (Any): The NPC object.
    Returns:
        Union[str, Dict[str, Any]]: The result of handling
        the jinx call.

    """
    if npc is None and team is None:
        return f"No jinxs are available. "
    else:

       
       
        if jinx_name not in npc.jinxs_dict and jinx_name not in team.jinxs_dict:
            print(f"Jinx {jinx_name} not available")
            if attempt < n_attempts:
                print(f"attempt {attempt+1} to generate jinx name failed, trying again")                
                return check_llm_command(
                                        f'''
                    In the previous attempt, the jinx name was: {jinx_name}.

                    That jinx was not available, only select those that are available.
                
                    If there are no available jinxs choose an alternative action. Do not invoke the jinx action. 


                    Here was the original command: BEGIN ORIGINAL COMMAND 
                    '''+ command +' END ORIGINAL COMMAND',
                    model=model,
                    provider=provider,
                    messages=messages,
                    npc=npc,
                    team=team,
                    stream=stream,
                    context=context
                )
            return {
                "output": f"Incorrect jinx name supplied and n_attempts reached.",
                "messages": messages,
            }




        elif jinx_name in npc.jinxs_dict:
            jinx = npc.jinxs_dict[jinx_name]
        elif jinx_name in team.jinxs_dict:
            jinx = team.jinxs_dict[jinx_name]

        render_markdown(f"jinx found: {jinx.jinx_name}")
        jinja_env = Environment(loader=FileSystemLoader("."), undefined=Undefined)
        example_format = {}
        for inp in jinx.inputs:
            if isinstance(inp, str):
                example_format[inp] = f"<value for {inp}>"
            elif isinstance(inp, dict):
                key = list(inp.keys())[0]
                example_format[key] = f"<value for {key}>"
        
        json_format_str = json.dumps(example_format, indent=4)
        

        prompt = f"""
        The user wants to use the jinx '{jinx_name}' with the following request:
        '{command}'"""


        prompt += f'Here were the previous 5 messages in the conversation: {messages[-5:]}'


        prompt+=f"""Here is the jinx file:
        ```
        {jinx.to_dict()}
        ```

        Please determine the required inputs for the jinx as a JSON object.
        
        They must be exactly as they are named in the jinx.
        For example, if the jinx has three inputs, you should respond with a list of three values that will pass for those args.
        
        If the jinx requires a file path, you must include an absolute path to the file including an extension.
        If the jinx requires code to be generated, you must generate it exactly according to the instructions.
        Your inputs must satisfy the jinx's requirements.




        Return only the JSON object without any markdown formatting.

        The format of the JSON object is: 
        
        """+"{"+json_format_str+"}"

        if npc and hasattr(npc, "shared_context"):
            if npc.shared_context.get("dataframes"):
                context_info = "\nAvailable dataframes:\n"
                for df_name in npc.shared_context["dataframes"].keys():
                    context_info += f"- {df_name}\n"
                prompt += f"""Here is contextual info that may affect your choice: {context_info}
                """                
        response = get_llm_response(
            prompt,
            format="json",
            model=model,
            provider=provider,
            messages=messages[-10:], 
            npc=npc,
            context=context
        )

        try:
            response_text = response.get("response", "{}")
            if isinstance(response_text, str):
                response_text = (
                    response_text.replace("```json", "").replace("```", "").strip()
                )

           
            if isinstance(response_text, dict):
                input_values = response_text
            else:
                input_values = json.loads(response_text)
           
        except json.JSONDecodeError as e:
            print(f"Error decoding input values: {e}. Raw response: {response}")
            return f"Error extracting inputs for jinx '{jinx_name}'"
       
        required_inputs = jinx.inputs
        missing_inputs = []
        for inp in required_inputs:
            if not isinstance(inp, dict):
               
                if inp not in input_values or input_values[inp] == "":
                    missing_inputs.append(inp)
        if len(missing_inputs) > 0:
           
            if attempt < n_attempts:
                print(f"attempt {attempt+1} to generate inputs failed, trying again")
                print("missing inputs", missing_inputs)
                print("llm response", response)
                print("input values", input_values)
                return handle_jinx_call(
                    command +' . In the previous attempt, the inputs were: ' + str(input_values) + ' and the missing inputs were: ' + str(missing_inputs) +' . Please ensure to not make this mistake again.',
                    jinx_name,
                    model=model,
                    provider=provider,
                    messages=messages,
                    npc=npc,
                    team=team,
                    stream=stream,
                    attempt=attempt + 1,
                    n_attempts=n_attempts,
                    context=context
                )
            return {
                "output": f"Missing inputs for jinx '{jinx_name}': {missing_inputs}",
                "messages": messages,
            }

        
        render_markdown( "\n".join(['\n - ' + str(key) + ': ' +str(val) for key, val in input_values.items()]))

        try:
            jinx_output = jinx.execute(
                input_values,
                jinja_env,
                npc=npc,
                messages=messages,
            )
        except Exception as e:
            print(f"An error occurred while executing the jinx: {e}")
            print(f"trying again, attempt {attempt+1}")
            print('command', command)
            if attempt < n_attempts:
                jinx_output = handle_jinx_call(
                    command,
                    jinx_name,
                    model=model,
                    provider=provider,
                    messages=messages,
                    npc=npc,
                    team=team,
                    stream=stream,
                    attempt=attempt + 1,
                    n_attempts=n_attempts,
                    context=f""" \n \n \n "jinx failed: {e}  \n \n \n here was the previous attempt: {input_values}""",
                )
        if not stream and len(messages) > 0 :            
            render_markdown(f""" ## jinx OUTPUT FROM CALLING {jinx_name} \n \n output:{jinx_output['output']}""" )            
            response = get_llm_response(f"""
                The user had the following request: {command}. 
                Here were the jinx outputs from calling {jinx_name}: {jinx_output}
                
                Given the jinx outputs and the user request, please format a simple answer that 
                provides the answer without requiring the user to carry out any further steps.
                """,
                model=model,
                provider=provider,
                npc=npc,
                messages=messages[-10:],
                context=context, 
                stream=stream,
            )
            messages = response['messages']
            response = response.get("response", {})
            return {'messages':messages, 'output':response}
        
        return {'messages': messages, 'output': jinx_output['output']}


def handle_request_input(
    context: str,
    model: str ,
    provider: str 
):
    """
    Analyze text and decide what to request from the user
    """
    prompt = f"""
    Analyze the text:
    {context}
    and determine what additional input is needed.
    Return a JSON object with:
    {{
        "input_needed": boolean,
        "request_reason": string explaining why input is needed,
        "request_prompt": string to show user if input needed
    }}

    Do not include any additional markdown formatting or leading ```json tags. Your response
    must be a valid JSON object.
    """

    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        messages=[],
        format="json",
    )

    result = response.get("response", {})
    if isinstance(result, str):
        result = json.loads(result)

    user_input = request_user_input(
        {"reason": result["request_reason"], "prompt": result["request_prompt"]},
    )
    return user_input





def jinx_handler(command, extracted_data, **kwargs):
    return handle_jinx_call(
        command, 
        extracted_data.get('jinx_name'),
        model=kwargs.get('model'),
        provider=kwargs.get('provider'),
        api_url=kwargs.get('api_url'),
        api_key=kwargs.get('api_key'),
        messages=kwargs.get('messages'),
        npc=kwargs.get('npc'),
        team = kwargs.get('team'),
        stream=kwargs.get('stream'),

        context=kwargs.get('context')
    )

def answer_handler(command, extracted_data, **kwargs):

    response =  get_llm_response(
        f"""
        
        Here is the user question: {command}
        
        
        Do not needlessly reference the user's files or provided context.
        
        Simply provide the answer to the user's question. Avoid
        appearing zany or unnecessarily forthcoming about the fact that you have received such information. You know it
        and the user knows it. there is no need to constantly mention the facts that are aware to both.
        
        Your previous commnets on this topic: {extracted_data.get('explanation', '')}
        
        """,
        model=kwargs.get('model'),
        provider=kwargs.get('provider'),
        api_url=kwargs.get('api_url'),
        api_key=kwargs.get('api_key'),
        messages=kwargs.get('messages',)[-10:],
        npc=kwargs.get('npc'),
        team=kwargs.get('team'), 
        stream=kwargs.get('stream', False),
        images=kwargs.get('images'), 
        context=kwargs.get('context')
    )
 
    return response
    
def check_llm_command(
    command: str,
    model: str = None,
    provider: str = None,
    api_url: str = None,
    api_key: str = None,
    npc: Any = None,
    team: Any = None,
    messages: List[Dict[str, str]] = None,
    images: list = None,
    stream=False,
    context=None,
    actions: Dict[str, Dict] = None,
):
    """This function checks an LLM command and returns sequences of steps with parallel actions."""
    if messages is None:
        messages = []

    if actions is None:
        actions = DEFAULT_ACTION_SPACE.copy()
    exec =  execute_multi_step_plan(
        command=command,
        model=model,
        provider=provider,
        api_url=api_url,
        api_key=api_key,
        npc=npc,
        team=team,
        messages=messages,
        images=images,
        stream=stream,
        context=context,
        actions=actions,

    )
    return exec




def jinx_context_filler(npc, team):
    """
    Generate context information about available jinxs for NPCs and teams.
    
    Args:
        npc: The NPC object
        team: The team object
    
    Returns:
        str: Formatted string containing jinx information and usage guidelines
    """
  
    npc_jinxs = "\nNPC Jinxs:\n" + (
        "\n".join(
            f"- {name}: {jinx.description}"
            for name, jinx in getattr(npc, "jinxs_dict", {}).items()
        )
        if getattr(npc, "jinxs_dict", None)
        else ''
    )
    
  
    team_jinxs = "\n\nTeam Jinxs:\n" + (
        "\n".join(
            f"- {name}: {jinx.description}"
            for name, jinx in getattr(team, "jinxs_dict", {}).items()
        )
        if team and getattr(team, "jinxs_dict", None)
        else ''
    )
    
  
    usage_guidelines = """
Use jinxs when appropriate. For example:

- If you are asked about something up-to-date or dynamic (e.g., latest exchange rates)
- If the user asks you to read or edit a file
- If the user asks for code that should be executed
- If the user requests to open, search, download or scrape, which involve actual system or web actions
- If they request a screenshot, audio, or image manipulation
- Situations requiring file parsing (e.g., CSV or JSON loading)
- Scripted workflows or pipelines, e.g., generate a chart, fetch data, summarize from source, etc.

You MUST use a jinx if the request directly refers to a tool the AI cannot handle directly (e.g., 'run', 'open', 'search', etc).

You must NOT use a jinx if:
- The user asks you to write them a story (unless they specify saving it to a file)
- To answer simple questions
- To determine general information that does not require up-to-date details
- To answer questions that can be answered with existing knowledge

To invoke a jinx, return the action 'invoke_jinx' along with the jinx specific name. 
An example for a jinx-specific return would be:
{
    "action": "invoke_jinx",
    "jinx_name": "file_reader",
    "explanation": "Read the contents of <full_filename_path_from_user_request> and <detailed explanation of how to accomplish the problem outlined in the request>."
}

Do not use the jinx names as the action keys. You must use the action 'invoke_jinx' to invoke a jinx!
Do not invent jinx names. Use only those provided.

Here are the currently available jinxs:"""


    

    if not npc_jinxs and not team_jinxs:
        return "No jinxs are available."
    else:    
        output = usage_guidelines
        if npc_jinxs:
            output += npc_jinxs
        if team_jinxs:
            output += team_jinxs
        return output
            
            
            
DEFAULT_ACTION_SPACE = {
    "invoke_jinx": {
        "description": "Invoke a jinx (jinja-template execution script)",
        "handler": jinx_handler,
        "context": lambda npc=None, team=None, **_: jinx_context_filler(npc, team),
        "output_keys": {
            "jinx_name": {
                "description": "The name of the jinx to invoke. must be from the provided list verbatim",
                "type": "string"
            }
        }
    },
    "answer": {
        "description": "Provide a direct informative answer",
        "handler": answer_handler,
        "context": """For general questions, use existing knowledge. For most queries a single action to answer a question will be sufficient.
e.g.

{
    "actions": [
        {
            "action": "answer",
            "explanation": "Provide a direct answer to the user's question based on existing knowledge."
            
    
        }
    ]
}

This should be preferred for more than half of requests. Do not overcomplicate the process.
Starting dialogue is usually more useful than using tools willynilly. Think carefully about 
the user's intent and use this action as an opportunity to clear up potential ambiguities before
proceeding to more complex actions.
For example, if a user requests to write a story, 
it is better to respond with 'answer'  and to write them a story rather than to invoke some tool.
Indeed, it might be even better to respond and to request clarification about what other elements they would liek to specify with the story.
Natural language is highly ambiguous and it is important to establish common ground and priorities before proceeding to more complex actions.

""",
        "output_keys": {}
    }
}
def plan_multi_step_actions(
   command: str,
   actions: Dict[str, Dict],
   npc: Any = None,
   team: Any = None,
   model: str = None,
   provider: str = None,
   api_url: str = None,
   api_key: str = None,
   context: str = None,
   messages: List[Dict[str, str]] = None,

):
    """
    Analyzes the user's command and creates a complete, sequential plan of actions
    by dynamically building a prompt from the provided action space.
    """
    
    
    prompt = f"""
Analyze the user's request: "{command}"

Your task is to create a complete, sequential JSON plan to fulfill the entire request.
Use the following context about available actions and tools to construct the plan.

"""
    if messages == None:
        messages = list()
    for action_name, action_info in actions.items():
        ctx = action_info.get("context")
        if callable(ctx):
            try:
              
                ctx = ctx(npc=npc, team=team)
            except Exception as e:
                print( actions)
                print(f"[WARN] Failed to render context for action '{action_name}': {e}")
                ctx = None
        
        if ctx:
            prompt += f"\n--- Context for action '{action_name}' ---\n{ctx}\n"
    if len(messages) >0:
        prompt += f'Here were the previous 5 messages in the conversation: {messages[-5:]}'

    prompt += f"""
--- Instructions ---
Based on the user's request and the context provided above, create a plan.

The plan must be a JSON object with a single key, "actions". Each action must include:
- "action": The name of the action to take.
- "explanation": A clear description of the goal for this specific step.
 
An Example Plan might look like this depending on the available actions:
""" + """
{
  "actions": [
    {
      "action": "<action_name_1>",
      "<action_specific_key_1..>": "<action_specific_value_1>",
      <...> : ...,      
      "explanation": "Identify the current CEO of Microsoft."
    },
    {
      "action": "<action_name_2>",
        "<action_specific_key_1..>": "<action_specific_value_1>",
        "explanation": "Find the <action-specific> information identified in the previous step."
    }
  ]
}

The plans should mostly be 1-2 actions and usually never more than 3 actions at a time.
Interactivity is important, unless a user specifies a usage of a specific action, it is generally best to
assume just to respond in the simplest way possible rather than trying to assume certain actions have been requested.

"""+f"""
Now, create the plan for the user's query: "{command}"
Respond ONLY with the plan.
"""

    action_response = get_llm_response(
        prompt,
        model=model, 
        provider=provider,
        api_url=api_url, 
        api_key=api_key,
        npc=npc,
        team=team,
        format="json", 
        messages=[], 
        context=context,
    )
    response_content = action_response.get("response", {})
  
  
    return response_content.get("actions", [])

def execute_multi_step_plan(
   command: str,
   model: str = None,
   provider: str = None,
   api_url: str = None,
   api_key: str = None,
   npc: Any = None,
   team: Any = None,
   messages: List[Dict[str, str]] = None,
   images: list = None,
   stream=False,
   context=None,

   actions: Dict[str, Dict] = None,
   **kwargs, 
):
    """
    Creates a comprehensive plan and executes it sequentially, passing context
    between steps for adaptive behavior.
    """
    
  
    planned_actions = plan_multi_step_actions(
        command=command,
        actions=actions,
        npc=npc,
        model=model,
        provider=provider,
        api_url=api_url,
        api_key=api_key,
        context=context,
        messages=messages,
        team=team, 
        
    )
    
    if not planned_actions:
        print("Could not generate a multi-step plan. Answering directly.")
        result = answer_handler(command=command, 
                                extracted_data={"explanation": "Answering the user's query directly."}, 
                                model=model,
                                provider=provider,
                                api_url=api_url,
                                api_key=api_key, 
                                messages=messages,
                                npc=npc,
                                stream=stream,
                                team = team, 
                                images=images, 
                                context=context)
        return {"messages": result.get('messages',
                                       messages), 
                "output": result.get('response')}


    step_outputs = []
    current_messages = messages.copy()
    render_markdown(f"### Plan for Command: {command[:100]}")
    for action in planned_actions:
        step_info = json.dumps({'action': action.get('action', ''), 
                                'explanation': str(action.get('explanation',''))[0:10]+'...'})
        render_markdown(f'- {step_info}')


    
    for i, action_data in enumerate(planned_actions):
        render_markdown(f"--- Executing Step {i + 1} of {len(planned_actions)} ---")
        action_name = action_data["action"]
      
          
        try:
            handler = actions[action_name]["handler"]


          
            step_context = f"Context from previous steps: {json.dumps(step_outputs)}" if step_outputs else ""
            render_markdown(
                f"- Executing Action: {action_name} \n- Explanation: {action_data.get('explanation')}\n "
            )
                
            result = handler(
                command=command, 
                extracted_data=action_data,
                model=model,
                provider=provider, 
                api_url=api_url,
                api_key=api_key, 
                messages=current_messages, 
                npc=npc,
                team=team,
                stream=stream, 

                context=context+step_context, 
                images=images
                )
        except KeyError as e:
          
            return execute_multi_step_plan(
                                            command=command + 'This error occurred: '+str(e)+'\n Do not make the same mistake again. If you are intending to use a jinx, you must `invoke_jinx`. If you just need to answer, choose `answer`.',
                                            model= model,
                                            provider = provider,
                                            api_url = api_url,
                                            api_key = api_key,
                                            npc = npc,
                                            team = team,
                                            messages = messages,
                                            images = images,
                                            stream=stream,
                                            context=context,
                                            actions=actions,
                                            **kwargs, 
            )

        action_output = result.get('output') or result.get('response')
        
        if stream and len(planned_actions) > 1:
          
            action_output = print_and_process_stream_with_markdown(action_output, model, provider)
        elif len(planned_actions) == 1:
          
          
            return {"messages": result.get('messages', 
                                           current_messages), 
                    "output": action_output}
        step_outputs.append(action_output)        
        current_messages = result.get('messages', 
                                      current_messages)

  
  
    final_output = compile_sequence_results(
       original_command=command,
       outputs=step_outputs,
       model=model, 
       provider=provider,
       npc=npc, 
       stream=stream, 
       context=context,
       **kwargs
    )
    
    return {"messages": current_messages, 
            "output": final_output}

def compile_sequence_results(original_command: str, 
                             outputs: List[str], 
                             model: str = None, 
                             provider: str = None, 
                             npc: Any = None, 
                             team: Any = None,
                             context: str = None,
                             stream: bool = False,
                             **kwargs) -> str:
    """
    Synthesizes a list of outputs from sequential steps into a single,
    coherent final response, framed as an answer to the original query.
    """
    if not outputs:
        return "The process completed, but produced no output."    
    synthesis_prompt = f"""
A user asked the following question:
"{original_command}"

To answer this, the following information was gathered in sequential steps:
{json.dumps(outputs, indent=2)}

Based *directly on the user's original question* and the information gathered, please
provide a single, final, and coherent response. Answer the user's question directly.
Do not mention the steps taken.

Final Synthesized Response that addresses the user in a polite and informative manner:
"""

    response = get_llm_response(
        synthesis_prompt,
        model=model, 
        provider=provider, 
        npc=npc, 
        team=team,
        messages=[], 
        stream=stream,
        context=context,
        **kwargs
    )
    synthesized = response.get("response", "")
    if synthesized:
        return synthesized    
    return '\n'.join(outputs)



def should_continue_with_more_actions(
    original_command: str,
    completed_actions: List[Dict[str, Any]],
    current_messages: List[Dict[str, str]],
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    context: str = None,
    **kwargs: Any
    
) -> Dict:
    """Decide if more action sequences are needed."""
    
    results_summary = ""
    for idx, action_result in enumerate(completed_actions):
        action_name = action_result.get("action", "Unknown Action")
        output = action_result.get('output', 'No Output')
        output_preview = output[:100] + "..." if isinstance(output, str) and len(output) > 100 else output
        results_summary += f"{idx + 1}. {action_name}: {output_preview}\n"

    prompt = f"""
Original user request: "{original_command}"

This request asks for multiple things. Analyze if ALL parts have been addressed.
Look for keywords like "and then", "use that to", "after that" which indicate multiple tasks.

Completed actions so far:
{results_summary}

For the request "{original_command}", identify:
1. What parts have been completed
2. What parts still need to be done

JSON response:
{{
    "needs_more_actions": true/false,
    "reasoning": "explain what's been done and what's still needed",
    "next_focus": "if more actions needed, what specific task should be done next"
}}
"""

    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        npc=npc,
        team=team,
        format="json",
        messages=[],
        
        context=context,
        **kwargs
    )
    
    response_dict = response.get("response", {})
    if not isinstance(response_dict, dict):
        return {"needs_more_actions": False, "reasoning": "Error", "next_focus": ""}
        
    return response_dict








def identify_groups(
    facts: List[str],
    model,
    provider,
    npc =  None,
    context: str = None,
    **kwargs
) -> List[str]:
    """Identify natural groups from a list of facts"""

        
    prompt = """What are the main groups these facts could be organized into?
    Express these groups in plain, natural language.

    For example, given:
        - User enjoys programming in Python
        - User works on machine learning projects
        - User likes to play piano
        - User practices meditation daily

    You might identify groups like:
        - Programming
        - Machine Learning
        - Musical Interests
        - Daily Practices

    Return a JSON object with the following structure:
        `{
            "groups": ["list of group names"]
        }`


    Return only the JSON object. Do not include any additional markdown formatting or
    leading json characters.
    """

    response = get_llm_response(
        prompt + f"\n\nFacts: {json.dumps(facts)}",
        model=model,
        provider=provider,
        format="json",
        npc=npc,
        context=context,
        
        **kwargs
    )
    return response["response"]["groups"]

def get_related_concepts_multi(node_name: str, 
                               node_type: str, 
                               all_concept_names, 
                               model: str = None,
                               provider: str = None,
                               npc=None,
                               context : str = None, 
                               **kwargs):
    """Links any node (fact or concept) to ALL relevant concepts in the entire ontology."""
    prompt = f"""
    Which of the following concepts from the entire ontology relate to the given {node_type}?
    Select all that apply, from the most specific to the most abstract.

    {node_type.capitalize()}: "{node_name}"

    Available Concepts:
    {json.dumps(all_concept_names, indent=2)}

    Respond with JSON: {{"related_concepts": ["Concept A", "Concept B", ...]}}
    """
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider, 
                                format="json", 
                                npc=npc,
                                context=context, 
                                **kwargs)
    return response["response"].get("related_concepts", [])


def assign_groups_to_fact(
    fact: str,
    groups: List[str],
    model = None,
    provider = None,
    npc = None, 
    context: str = None,
    **kwargs
) -> Dict[str, List[str]]:
    """Assign facts to the identified groups"""
    prompt = f"""Given this fact, assign it to any relevant groups.

    A fact can belong to multiple groups if it fits.

    Here is the fact: {fact}

    Here are the groups: {groups}

    Return a JSON object with the following structure:
        {{
            "groups": ["list of group names"]
        }}

    Do not include any additional markdown formatting or leading json characters.


    """

    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        format="json",
        npc=npc,
        context=context,
        **kwargs
    )
    return response["response"]

def generate_group_candidates(
    items: List[str],
    item_type: str,
    model: str = None,
    provider: str =None,
    npc = None,
    context: str = None,
    n_passes: int = 3,
    subset_size: int = 10, 
    **kwargs
) -> List[str]:
    """Generate candidate groups for items (facts or groups) based on core semantic meaning."""
    all_candidates = []
    
    for pass_num in range(n_passes):
        if len(items) > subset_size:
            item_subset = random.sample(items, min(subset_size, len(items)))
        else:
            item_subset = items
        
      
        prompt = f"""From the following {item_type}, identify specific and relevant conceptual groups.
        Think about the core subject or entity being discussed.
        
        GUIDELINES FOR GROUP NAMES:
        1.  **Prioritize Specificity:** Names should be precise and directly reflect the content.
        2.  **Favor Nouns and Noun Phrases:** Use descriptive nouns or noun phrases.
        3.  **AVOID:**
            *   Gerunds (words ending in -ing when used as nouns, like "Understanding", "Analyzing", "Processing"). If a gerund is unavoidable, try to make it a specific action (e.g., "User Authentication Module" is better than "Authenticating Users").
            *   Adverbs or descriptive adjectives that don't form a core part of the subject's identity (e.g., "Quickly calculating", "Effectively managing").
            *   Overly generic terms (e.g., "Concepts", "Processes", "Dynamics", "Mechanics", "Analysis", "Understanding", "Interactions", "Relationships", "Properties", "Structures", "Systems", "Frameworks", "Predictions", "Outcomes", "Effects", "Considerations", "Methods", "Techniques", "Data", "Theoretical", "Physical", "Spatial", "Temporal").
        4.  **Direct Naming:** If an item is a specific entity or action, it can be a group name itself (e.g., "Earth", "Lamb Shank Braising", "World War I").
        
        EXAMPLE:
        Input {item_type.capitalize()}: ["Self-intersection shocks drive accretion disk formation.", "Gravity stretches star into stream.", "Energy dissipation in shocks influences capture fraction."]
        Desired Output Groups: ["Accretion Disk Formation (Self-Intersection Shocks)", "Stellar Tidal Stretching", "Energy Dissipation from Shocks"]
        
        ---
        
        Now, analyze the following {item_type}:
        {item_type.capitalize()}: {json.dumps(item_subset)}
        
        Return a JSON object:
        {{
            "groups": ["list of specific, precise, and relevant group names"]
        }}
        """
      
        
        response = get_llm_response(
            prompt,
            model=model,
            provider=provider,
            format="json",
            npc=npc,
            context=context,
            **kwargs
        )
        
        candidates = response["response"].get("groups", [])
        all_candidates.extend(candidates)

    return list(set(all_candidates))


def remove_idempotent_groups(
    group_candidates: List[str],
    model: str = None,
    provider: str =None,
    npc = None, 
    context : str = None,
    **kwargs: Any
) -> List[str]:
    """Remove groups that are essentially identical in meaning, favoring specificity and direct naming, and avoiding generic structures."""
    
    prompt = f"""Compare these group names. Identify and list ONLY the groups that are conceptually distinct and specific.
    
    GUIDELINES FOR SELECTING DISTINCT GROUPS:
    1.  **Prioritize Specificity and Direct Naming:** Favor precise nouns or noun phrases that directly name the subject.
    2.  **Prefer Concrete Entities/Actions:** If a name refers to a specific entity or action (e.g., "Earth", "Sun", "Water", "France", "User Authentication Module", "Lamb Shank Braising", "World War I"), keep it if it's distinct.
    3.  **Rephrase Gerunds:** If a name uses a gerund (e.g., "Understanding TDEs"), rephrase it to a noun or noun phrase (e.g., "Tidal Disruption Events").
    4.  **AVOID OVERLY GENERIC TERMS:** Do NOT use very broad or abstract terms that don't add specific meaning. Examples to avoid: "Concepts", "Processes", "Dynamics", "Mechanics", "Analysis", "Understanding", "Interactions", "Relationships", "Properties", "Structures", "Systems", "Frameworks", "Predictions", "Outcomes", "Effects", "Considerations", "Methods", "Techniques", "Data", "Theoretical", "Physical", "Spatial", "Temporal". If a group name seems overly generic or abstract, it should likely be removed or refined.
    5.  **Similarity Check:** If two groups are very similar, keep the one that is more descriptive or specific to the domain.

    EXAMPLE 1:
    Groups: ["Accretion Disk Formation", "Accretion Disk Dynamics", "Formation of Accretion Disks"]
    Distinct Groups: ["Accretion Disk Formation", "Accretion Disk Dynamics"] 

    EXAMPLE 2:
    Groups: ["Causes of Events", "Event Mechanisms", "Event Drivers"]
    Distinct Groups: ["Event Causation", "Event Mechanisms"] 

    EXAMPLE 3:
    Groups: ["Astrophysics Basics", "Fundamental Physics", "General Science Concepts"]
    Distinct Groups: ["Fundamental Physics"] 

    EXAMPLE 4:
    Groups: ["Earth", "The Planet Earth", "Sun", "Our Star"]
    Distinct Groups: ["Earth", "Sun"]
    
    EXAMPLE 5:
    Groups: ["User Authentication Module", "Authentication System", "Login Process"]
    Distinct Groups: ["User Authentication Module", "Login Process"]
    
    ---
    
    Now, analyze the following groups:
    Groups: {json.dumps(group_candidates)}
    
    Return JSON:
    {{
        "distinct_groups": ["list of specific, precise, and distinct group names to keep"]
    }}
    """
    
    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        format="json",
        npc=npc,
        context=context,
        **kwargs
    )
    
    return response["response"]["distinct_groups"]

def breathe(
    messages: List[Dict[str, str]],
    model: str = None,
    provider: str = None, 
    npc =  None,
    context: str = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Condense the conversation context into a small set of key extractions."""
    if not messages:
        return {"output": {}, "messages": []}

    if 'stream' in kwargs:
        kwargs['stream'] = False
    conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])


    prompt = f'''
    Read the following conversation:

    {conversation_text}

    ''' +'''

    Now identify the following items:

    1. The high level objective
    2. The most recent task
    3. The accomplishments thus far
    4. The failures thus far


    Return a JSON like so:

    {
        "high_level_objective": "the overall goal so far for the user", 
        "most_recent_task": "The currently ongoing task", 
        "accomplishments": ["accomplishment1", "accomplishment2"], 
        "failures": ["falures1", "failures2"], 
    }

    '''

    
    result = get_llm_response(prompt, 
                           model=model, 
                           provider=provider, 
                           npc=npc, 
                           context=context, 
                           format='json', 
                           **kwargs)

    res = result.get('response', {})
    if isinstance(res, str):
        raise Exception
    format_output = f"""Here is a summary of the previous session. 
    The high level objective was: {res.get('high_level_objective')} \n The accomplishments were: {res.get('accomplishments')}, 
    the failures were: {res.get('failures')} and the most recent task was: {res.get('most_recent_task')}   """
    return {'output': format_output, 
            'messages': [
                         {
                           'content': format_output, 
                           'role': 'assistant'}
                           ] 
                          }
def abstract(groups, 
             model, 
             provider, 
             npc=None,
             context: str = None, 
             **kwargs):
    """
    Create more abstract terms from groups.
    """
    sample_groups = random.sample(groups, min(len(groups), max(3, len(groups) // 2)))
    
    groups_text_for_prompt = "\n".join([f'- "{g["name"]}"' for g in sample_groups])

    prompt = f"""
        Create more abstract categories from this list of groups.

        Groups:
        {groups_text_for_prompt}

        You will create higher-level concepts that interrelate between the given groups. 

        Create abstract categories that encompass multiple related facts, but do not unnecessarily combine facts with conjunctions. For example, do not try to combine "characters", "settings", and "physical reactions" into a
        compound group like "Characters, Setting, and Physical Reactions". This kind of grouping is not productive and only obfuscates true abstractions. 
        For example, a group that might encompass the three aforermentioned names might be "Literary Themes" or "Video Editing Functionis", depending on the context.
        Your aim is to abstract, not to just arbitrarily generate associations. 

        Group names should never be more than two words. They should not contain gerunds. They should never contain conjunctions like "AND" or "OR".
        Generate no more than 5 new concepts and no fewer than 2. 

        Respond with JSON:
        {{
            "groups": [
                {{
                    "name": "abstract category name"
                }}
            ]
        }}
        """
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider, 
                                format="json",
                                npc=npc,
                                context=context, 
                                **kwargs)

    return response["response"].get("groups", [])


def extract_facts(
    text: str,
    model: str,
    provider: str,
    npc = None,
    context: str = None
) -> List[str]:
    """Extract concise facts from text using LLM (as defined earlier)"""
  
    prompt = """Extract concise facts from this text.
        A fact is a piece of information that makes a statement about the world.
        A fact is typically a sentence that is true or false.
        Facts may be simple or complex. They can also be conflicting with each other, usually
        because there is some hidden context that is not mentioned in the text.
        In any case, it is simply your job to extract a list of facts that could pertain to
        an individual's personality.
        
        For example, if a message says:
            "since I am a doctor I am often trying to think up new ways to help people.
            Can you help me set up a new kind of software to help with that?"
        You might extract the following facts:
            - The individual is a doctor
            - They are helpful

        Another example:
            "I am a software engineer who loves to play video games. I am also a huge fan of the
            Star Wars franchise and I am a member of the 501st Legion."
        You might extract the following facts:
            - The individual is a software engineer
            - The individual loves to play video games
            - The individual is a huge fan of the Star Wars franchise
            - The individual is a member of the 501st Legion

        Another example:
            "The quantum tunneling effect allows particles to pass through barriers
            that classical physics says they shouldn't be able to cross. This has
            huge implications for semiconductor design."
        You might extract these facts:
            - Quantum tunneling enables particles to pass through barriers that are
              impassable according to classical physics
            - The behavior of quantum tunneling has significant implications for
              how semiconductors must be designed

        Another example:
            "People used to think the Earth was flat. Now we know it's spherical,
            though technically it's an oblate spheroid due to its rotation."
        You might extract these facts:
            - People historically believed the Earth was flat
            - It is now known that the Earth is an oblate spheroid
            - The Earth's oblate spheroid shape is caused by its rotation

        Another example:
            "My research on black holes suggests they emit radiation, but my professor
            says this conflicts with Einstein's work. After reading more papers, I
            learned this is actually Hawking radiation and doesn't conflict at all."
        You might extract the following facts:
            - Black holes emit radiation
            - The professor believes this radiation conflicts with Einstein's work
            - The radiation from black holes is called Hawking radiation
            - Hawking radiation does not conflict with Einstein's work

        Another example:
            "During the pandemic, many developers switched to remote work. I found
            that I'm actually more productive at home, though my company initially
            thought productivity would drop. Now they're keeping remote work permanent."
        You might extract the following facts:
            - The pandemic caused many developers to switch to remote work
            - The individual discovered higher productivity when working from home
            - The company predicted productivity would decrease with remote work
            - The company decided to make remote work a permanent option

        Thus, it is your mission to reliably extract lists of facts.

        Return a JSON object with the following structure:
            {
                "fact_list": "a list containing the facts where each fact is a string",
            }
    """ 
    if context and len(context) > 0:
        prompt+=f""" Here is some relevant user context: {context}"""

    prompt+="""    
    Return only the JSON object.
    Do not include any additional markdown formatting.
    """

    response = get_llm_response(
        prompt + f"HERE BEGINS THE TEXT TO INVESTIGATE:\n\nText: {text}",
        model=model,
        provider=provider,
        format="json",
        npc=npc,
        context=context,
    )
    response = response["response"]
    return response.get("fact_list", [])


def get_facts(content_text, 
              model= None,
              provider = None,
              npc=None,
              context : str=None, 
              attempt_number=1,
              n_attempts=3,

              **kwargs):
    """Extract facts from content text"""
    
    prompt = f"""
    Extract facts from this text. A fact is a specific statement that can be sourced from the text.

    Example: if text says "the moon is the earth's only currently known satellite", extract:
    - "The moon is a satellite of earth" 
    - "The moon is the only current satellite of earth"
    - "There may have been other satellites of earth" (inferred from "only currently known")


        A fact is a piece of information that makes a statement about the world.
        A fact is typically a sentence that is true or false.
        Facts may be simple or complex. They can also be conflicting with each other, usually
        because there is some hidden context that is not mentioned in the text.
        In any case, it is simply your job to extract a list of facts that could pertain to
        an individual's personality.
        
        For example, if a message says:
            "since I am a doctor I am often trying to think up new ways to help people.
            Can you help me set up a new kind of software to help with that?"
        You might extract the following facts:
            - The individual is a doctor
            - They are helpful

        Another example:
            "I am a software engineer who loves to play video games. I am also a huge fan of the
            Star Wars franchise and I am a member of the 501st Legion."
        You might extract the following facts:
            - The individual is a software engineer
            - The individual loves to play video games
            - The individual is a huge fan of the Star Wars franchise
            - The individual is a member of the 501st Legion

        Another example:
            "The quantum tunneling effect allows particles to pass through barriers
            that classical physics says they shouldn't be able to cross. This has
            huge implications for semiconductor design."
        You might extract these facts:
            - Quantum tunneling enables particles to pass through barriers that are
              impassable according to classical physics
            - The behavior of quantum tunneling has significant implications for
              how semiconductors must be designed

        Another example:
            "People used to think the Earth was flat. Now we know it's spherical,
            though technically it's an oblate spheroid due to its rotation."
        You might extract these facts:
            - People historically believed the Earth was flat
            - It is now known that the Earth is an oblate spheroid
            - The Earth's oblate spheroid shape is caused by its rotation

        Another example:
            "My research on black holes suggests they emit radiation, but my professor
            says this conflicts with Einstein's work. After reading more papers, I
            learned this is actually Hawking radiation and doesn't conflict at all."
        You might extract the following facts:
            - Black holes emit radiation
            - The professor believes this radiation conflicts with Einstein's work
            - The radiation from black holes is called Hawking radiation
            - Hawking radiation does not conflict with Einstein's work

        Another example:
            "During the pandemic, many developers switched to remote work. I found
            that I'm actually more productive at home, though my company initially
            thought productivity would drop. Now they're keeping remote work permanent."
        You might extract the following facts:
            - The pandemic caused many developers to switch to remote work
            - The individual discovered higher productivity when working from home
            - The company predicted productivity would decrease with remote work
            - The company decided to make remote work a permanent option

        Thus, it is your mission to reliably extract lists of facts.

    Here is the text:
    Text: "{content_text}"

    Facts should never be more than one or two sentences, and they should not be overly complex or literal. They must be explicitly
    derived or inferred from the source text. Do not simply repeat the source text verbatim when stating the fact. 
    
    No two facts should share substantially similar claims. They should be conceptually distinct and pertain to distinct ideas, avoiding lengthy convoluted or compound facts .
    Respond with JSON:
    {{
        "facts": [
            {{
                "statement": "fact statement that builds on input text to state a specific claim that can be falsified through reference to the source material",
                "source_text": "text snippets related to the source text",
                "type": "explicit or inferred"
            }} 
        ]
    }}
    """
    
    response = get_llm_response(prompt, 
                                model=model,
                                provider=provider, 
                                npc=npc,
                                format="json", 
                                context=context,
                                **kwargs)

    if len(response.get("response", {}).get("facts", [])) == 0 and attempt_number < n_attempts:
        print(f"  Attempt {attempt_number} to extract facts yielded no results. Retrying...")
        return get_facts(content_text, 
                         model=model, 
                         provider=provider, 
                         npc=npc,
                         context=context,
                         attempt_number=attempt_number+1,
                         n_attempts=n_attempts,
                         **kwargs)
    
    return response["response"].get("facts", [])

        

def zoom_in(facts, 
            model= None,
            provider=None, 
            npc=None,
            context: str = None, 
            attempt_number: int = 1,
            n_attempts=3,            
            **kwargs):
    """Infer new implied facts from existing facts"""
    valid_facts = []
    for fact in facts:
        if isinstance(fact, dict) and 'statement' in fact:
            valid_facts.append(fact)
    if not valid_facts:
        return []     

    fact_lines = []
    for fact in valid_facts:
        fact_lines.append(f"- {fact['statement']}")
    facts_text = "\n".join(fact_lines)
    
    prompt = f"""
    Look at these facts and infer new implied facts:

    {facts_text}

    What other facts can be reasonably inferred from these?
    """ +"""
    Respond with JSON:
    {
        "implied_facts": [
            {
                "statement": "new implied fact",
                "inferred_from": ["which facts this comes from"]
            }
        ]
    }
    """
    
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider, 
                                format="json", 
                                context=context,
                                npc=npc,
                                **kwargs)

    facts =  response.get("response", {}).get("implied_facts", [])
    if len(facts) == 0:
        return zoom_in(valid_facts, 
                       model=model, 
                       provider=provider, 
                       npc=npc,
                       context=context,
                       attempt_number=attempt_number+1,
                       n_tries=n_tries,
                       **kwargs)
    return facts
def generate_groups(facts, 
                    model=None,
                    provider=None,
                    npc=None,
                    context: str =None, 
                    **kwargs):
    """Generate conceptual groups for facts"""
    
    facts_text = "\n".join([f"- {fact['statement']}" for fact in facts])
    
    prompt = f"""
    Generate conceptual groups for this group off facts:

    {facts_text}

    Create categories that encompass multiple related facts, but do not unnecessarily combine facts with conjunctions. 
    
    Your aim is to generalize commonly occurring ideas into groups, not to just arbitrarily generate associations. 
    Focus on the key commonly occurring items and expresions.     

    Group names should never be more than two words. They should not contain gerunds. They should never contain conjunctions like "AND" or "OR".
    Respond with JSON:
    {{
        "groups": [
            {{
                "name": "group name"
            }}
        ]
    }}
    """
    
    response = get_llm_response(prompt,
                                model=model, 
                                provider=provider, 
                                format="json", 
                                context=context,
                                npc=npc,
                                **kwargs)

    return response["response"].get("groups", [])

def remove_redundant_groups(groups, 
                            model=None,
                            provider=None,
                            npc=None,
                            context: str = None,
                            **kwargs):
    """Remove redundant groups"""
    
    groups_text = "\n".join([f"- {g['name']}" for g in groups])
    
    prompt = f"""
    Remove redundant groups from this list:

    {groups_text}



    Merge similar groups and keep only distinct concepts.
    Create abstract categories that encompass multiple related facts, but do not unnecessarily combine facts with conjunctions. For example, do not try to combine "characters", "settings", and "physical reactions" into a
    compound group like "Characters, Setting, and Physical Reactions". This kind of grouping is not productive and only obfuscates true abstractions. 
    For example, a group that might encompass the three aforermentioned names might be "Literary Themes" or "Video Editing Functionis", depending on the context.
    Your aim is to abstract, not to just arbitrarily generate associations. 

    Group names should never be more than two words. They should not contain gerunds. They should never contain conjunctions like "AND" or "OR".


    Respond with JSON:
    {{
        "groups": [
            {{
                "name": "final group name"
            }}
        ]
    }}
    """
    
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider, 
                                format="json", 
                                npc=npc,
                                context=context,
                                **kwargs)

    return response["response"].get("groups", [])


def prune_fact_subset_llm(fact_subset, 
                          concept_name, 
                          model=None,
                          provider=None,
                          npc=None,
                          context : str = None,
                          **kwargs):
    """Identifies redundancies WITHIN a small, topically related subset of facts."""
    print(f"  Step Sleep-A: Pruning fact subset for concept '{concept_name}'...")
    

    prompt = f"""
    The following facts are all related to the concept "{concept_name}".
    Review ONLY this subset and identify groups of facts that are semantically identical.
    Return only the set of facts that are semantically distinct, and archive the rest.

    Fact Subset: {json.dumps(fact_subset, indent=2)}

    Return a json list of groups 
    {{
        "refined_facts": [
            fact1,
            fact2,
            fact3,... 
        ]
    }}
    """
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider, 
                                npc=None,
                                format="json", 
                                context=context)
    return response['response'].get('refined_facts', [])

def consolidate_facts_llm(new_fact, 
                          existing_facts, 
                          model, 
                          provider, 
                          npc=None,
                          context: str =None,
                          **kwargs):
    """
    Uses an LLM to decide if a new fact is novel or redundant.
    """
    prompt = f"""
        Analyze the "New Fact" in the context of the "Existing Facts" list.
        Your task is to determine if the new fact provides genuinely new information or if it is essentially a repeat or minor rephrasing of information already present.

        New Fact:
        "{new_fact['statement']}"

        Existing Facts:
        {json.dumps([f['statement'] for f in existing_facts], indent=2)}

        Possible decisions:
        - 'novel': The fact introduces new, distinct information not covered by the existing facts.
        - 'redundant': The fact repeats information already present in the existing facts.

        Respond with a JSON object:
        {{
            "decision": "novel or redundant",
            "reason": "A brief explanation for your decision."
        }}
        """
    response = get_llm_response(prompt,
                                model=model, 
                                provider=provider, 
                                format="json", 
                                npc=npc,
                                context=context,
                                **kwargs)
    return response['response']


def get_related_facts_llm(new_fact_statement, 
                          existing_fact_statements, 
                          model = None, 
                          provider = None,
                          npc = None, 
                          attempt_number = 1,
                          n_attempts = 3,
                          context='', 
                          **kwargs):
    """Identifies which existing facts are causally or thematically related to a new fact."""
    prompt = f"""
    A new fact has been learned: "{new_fact_statement}"

    Which of the following existing facts are directly related to it (causally, sequentially, or thematically)?
    Select only the most direct and meaningful connections.

    Existing Facts:
    {json.dumps(existing_fact_statements, indent=2)}

    Respond with JSON: {{"related_facts": ["statement of a related fact", ...]}}
    """
    response = get_llm_response(prompt,
                                model=model, 
                                provider=provider, 
                                format="json", 
                                npc=npc,
                                context=context,
                                **kwargs)   
    if attempt_number > n_attempts:
        print(f"  Attempt {attempt_number} to find related facts yielded no results. Giving up.")
        return get_related_facts_llm(new_fact_statement, 
                                       existing_fact_statements, 
                                       model=model, 
                                       provider=provider, 
                                       npc=npc,
                                       attempt_number=attempt_number+1,
                                       n_attempts=n_attempts,
                                       context=context,
                                       **kwargs)    

    return response["response"].get("related_facts", [])

def find_best_link_concept_llm(candidate_concept_name, 
                               existing_concept_names, 
                               model = None,
                               provider = None,
                               npc = None,
                               context: str = None,
                               **kwargs   ):
    """
    Finds the best existing concept to link a new candidate concept to.
    This prompt now uses neutral "association" language.
    """
    prompt = f"""
    Here is a new candidate concept: "{candidate_concept_name}"
    
    Which of the following existing concepts is it most closely related to? The relationship could be as a sub-category, a similar idea, or a related domain.

    Existing Concepts:
    {json.dumps(existing_concept_names, indent=2)}

    Respond with the single best-fit concept to link to from the list, or respond with "none" if it is a genuinely new root idea.
    {{
      "best_link_concept": "The single best concept name OR none"
    }}
    """
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider, 
                                format="json", 
                                npc=npc,
                                context=context,
                                **kwargs)
    return response['response'].get('best_link_concept')

def asymptotic_freedom(parent_concept, 
                       supporting_facts, 
                       model=None, 
                       provider=None, 
                       npc = None,
                       context: str = None, 
                       **kwargs):
    """Given a concept and its facts, proposes an intermediate layer of sub-concepts."""
    print(f"  Step Sleep-B: Attempting to deepen concept '{parent_concept['name']}'...")
    fact_statements = []
    for f in supporting_facts:
        fact_statements.append(f['statement'])
        
    prompt = f"""
    The concept "{parent_concept['name']}" is supported by many diverse facts.
    Propose a layer of 2-4 more specific sub-concepts to better organize these facts.
    These new concepts will exist as nodes that link to "{parent_concept['name']}".

    Supporting Facts: {json.dumps(fact_statements, indent=2)}
    Respond with JSON: {{
        "new_sub_concepts": ["sub_layer1", "sub_layer2"]
    }}
    """
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider,
                                format="json", 
                                context=context, npc=npc,
                                **kwargs)
    return response['response'].get('new_sub_concepts', [])



def bootstrap(
    prompt: str,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    sample_params: Dict[str, Any] = None,
    sync_strategy: str = "consensus",
    context: str = None,
    n_samples: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """Bootstrap by sampling multiple agents from team or varying parameters"""
    
    if team and hasattr(team, 'npcs') and len(team.npcs) >= n_samples:
      
        sampled_npcs = list(team.npcs.values())[:n_samples]
        results = []
        
        for i, agent in enumerate(sampled_npcs):
            response = get_llm_response(
                f"Sample {i+1}: {prompt}\nContext: {context}",
                npc=agent,
                context=context,
                **kwargs
            )
            results.append({
                'agent': agent.name,
                'response': response.get("response", "")
            })
    else:
      
        if sample_params is None:
            sample_params = {"temperature": [0.3, 0.7, 1.0]}
        
        results = []
        for i in range(n_samples):
            temp = sample_params.get('temperature', [0.7])[i % len(sample_params.get('temperature', [0.7]))]
            response = get_llm_response(
                f"Sample {i+1}: {prompt}\nContext: {context}",
                model=model,
                provider=provider,
                npc=npc,
                temperature=temp,
                context=context,
                **kwargs
            )
            results.append({
                'variation': f'temp_{temp}',
                'response': response.get("response", "")
            })
    
  
    response_texts = [r['response'] for r in results]
    return synthesize(response_texts, sync_strategy, model, provider, npc or (team.forenpc if team else None), context)

def harmonize(
    prompt: str,
    items: List[str],
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    harmony_rules: List[str] = None,
    context: str = None,
    agent_roles: List[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Harmonize using multiple specialized agents"""
    
    if team and hasattr(team, 'npcs'):
      
        available_agents = list(team.npcs.values())
        
        if agent_roles:
          
            selected_agents = []
            for role in agent_roles:
                matching_agent = next((a for a in available_agents if role.lower() in a.name.lower() or role.lower() in a.primary_directive.lower()), None)
                if matching_agent:
                    selected_agents.append(matching_agent)
            agents_to_use = selected_agents or available_agents[:len(items)]
        else:
          
            agents_to_use = available_agents[:min(len(items), len(available_agents))]
        
        harmonized_results = []
        for i, (item, agent) in enumerate(zip(items, agents_to_use)):
            harmony_prompt = f"""Harmonize this element: {item}
Task: {prompt}
Rules: {', '.join(harmony_rules or ['maintain_consistency'])}
Context: {context}
Your role in harmony: {agent.primary_directive}"""
            
            response = get_llm_response(
                harmony_prompt,
                npc=agent,
                context=context,
                **kwargs
            )
            harmonized_results.append({
                'agent': agent.name,
                'item': item,
                'harmonized': response.get("response", "")
            })
        
      
        coordinator = team.get_forenpc() if team else npc
        synthesis_prompt = f"""Synthesize these harmonized elements:
{chr(10).join([f"{r['agent']}: {r['harmonized']}" for r in harmonized_results])}
Create unified harmonious result."""
        
        return get_llm_response(synthesis_prompt, npc=coordinator, context=context, **kwargs)
    
    else:
      
        items_text = chr(10).join([f"{i+1}. {item}" for i, item in enumerate(items)])
        harmony_prompt = f"""Harmonize these items: {items_text}
Task: {prompt}
Rules: {', '.join(harmony_rules or ['maintain_consistency'])}
Context: {context}"""
        
        return get_llm_response(harmony_prompt, model=model, provider=provider, npc=npc, context=context, **kwargs)

def orchestrate(
    prompt: str,
    items: List[str],
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    workflow: str = "sequential_coordination",
    context: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Orchestrate using team.orchestrate method"""
    
    if team and hasattr(team, 'orchestrate'):
      
        orchestration_request = f"""Orchestrate workflow: {workflow}
Task: {prompt}
Items: {chr(10).join([f'- {item}' for item in items])}
Context: {context}"""
        
        return team.orchestrate(orchestration_request)
    
    else:
      
        items_text = chr(10).join([f"{i+1}. {item}" for i, item in enumerate(items)])
        orchestrate_prompt = f"""Orchestrate using {workflow}:
Task: {prompt}
Items: {items_text}
Context: {context}"""
        
        return get_llm_response(orchestrate_prompt, model=model, provider=provider, npc=npc, context=context, **kwargs)

def spread_and_sync(
    prompt: str,
    variations: List[str],
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    sync_strategy: str = "consensus",
    context: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Spread across agents/variations then sync with distribution analysis"""
    
    if team and hasattr(team, 'npcs') and len(team.npcs) >= len(variations):
      
        agents = list(team.npcs.values())[:len(variations)]
        results = []
        
        for variation, agent in zip(variations, agents):
            variation_prompt = f"""Analyze from {variation} perspective:
Task: {prompt}
Context: {context}
Apply your expertise with {variation} approach."""
            
            response = get_llm_response(variation_prompt, npc=agent, context=context, **kwargs)
            results.append({
                'agent': agent.name,
                'variation': variation,
                'response': response.get("response", "")
            })
    else:
      
        results = []
        agent = npc or (team.get_forenpc() if team else None)
        
        for variation in variations:
            variation_prompt = f"""Analyze from {variation} perspective:
Task: {prompt}
Context: {context}"""
            
            response = get_llm_response(variation_prompt, model=model, provider=provider, npc=agent, context=context, **kwargs)
            results.append({
                'variation': variation,
                'response': response.get("response", "")
            })
    
  
    response_texts = [r['response'] for r in results]
    return synthesize(response_texts, sync_strategy, model, provider, npc or (team.get_forenpc() if team else None), context)

def criticize(
    prompt: str,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    context: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Provide critical analysis and constructive criticism"""
    critique_prompt = f"""
    Provide a critical analysis and constructive criticism of the following:
    {prompt}
    
    Focus on identifying weaknesses, potential improvements, and alternative approaches.
    Be specific and provide actionable feedback.
    """
    
    return get_llm_response(
        critique_prompt,
        model=model,
        provider=provider,
        npc=npc,
        team=team,
        context=context,
        **kwargs
    )
def synthesize(
    prompt: str,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    context: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Synthesize information from multiple sources or perspectives"""
    
    # Extract responses from kwargs if provided, otherwise use prompt as single response
    responses = kwargs.get('responses', [prompt])
    sync_strategy = kwargs.get('sync_strategy', 'consensus')
    
    # If we have multiple responses, create a synthesis prompt
    if len(responses) > 1:
        synthesis_prompt = f"""Synthesize these multiple perspectives:
        
        {chr(10).join([f'Response {i+1}: {r}' for i, r in enumerate(responses)])}
        
        Synthesis strategy: {sync_strategy}
        Context: {context}
        
        Create a coherent synthesis that incorporates key insights from all perspectives."""
    else:
        # For single response, just summarize/refine it
        synthesis_prompt = f"""Refine and synthesize this content:
        
        {responses[0]}
        
        Context: {context}
        
        Create a clear, concise synthesis that captures the essence of the content."""
    
    return get_llm_response(
        synthesis_prompt,
        model=model,
        provider=provider,
        npc=npc,
        team=team,
        context=context,
        **kwargs
    )