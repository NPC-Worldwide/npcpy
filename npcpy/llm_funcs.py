from jinja2 import Environment, FileSystemLoader, Undefined
import json
import os
import PIL
import random
import subprocess
import copy
import itertools
import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger("npcpy.llm_funcs")
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
    api_key: str = None,
):
    """This function generates an image using the specified provider and model.
    Args:
        prompt (str): The prompt for generating the image.
    Keyword Args:
        model (str): The model to use for generating the image.
        provider (str): The provider to use for generating the image.
        filename (str): The filename to save the image to.
        npc (Any): The NPC object.
        api_key (str): The API key for the image generation service.
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
        api_key=api_key,
        
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
                model=model,
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
    n_samples: int = 1,
    matrix: Optional[Dict[str, List[Any]]] = None,
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
    import logging
    logger = logging.getLogger("npcpy.llm_funcs")
    logger.debug(f"[get_llm_response] Received {len(messages) if messages else 0} messages, prompt_len={len(prompt) if prompt else 0}")

    def _resolve_model_provider(npc_override, team_override, model_override, provider_override):
        m = model_override
        p = provider_override
        a_url = api_url
        if m is not None and p is not None:
            pass
        elif p is None and m is not None:
            p = lookup_provider(m)
        elif npc_override is not None:
            if npc_override.provider is not None:
                p = npc_override.provider
            if npc_override.model is not None:
                m = npc_override.model
            if npc_override.api_url is not None:
                a_url = npc_override.api_url
        elif team_override is not None:
            if team_override.model is not None:
                m = team_override.model
            if team_override.provider is not None:
                p = team_override.provider
            if team_override.api_url is not None:
                a_url = team_override.api_url
        else:
            p = "ollama"
            if images is not None or attachments is not None:
                m = "llava:7b"
            else:
                m = "llama3.2"
        return m, p, a_url

    def _context_suffix(ctx):
        if ctx is not None:
            return f'\n\n\nUser Provided Context: {ctx}'
        return ''

    def _build_messages(base_messages, sys_msg, prompt_text, ctx_suffix):
        msgs = copy.deepcopy(base_messages) if base_messages else []
        if not msgs:
            msgs = [{"role": "system", "content": sys_msg}]
            if prompt_text:
                msgs.append({"role": "user", "content": prompt_text + ctx_suffix})
        elif prompt_text and msgs[-1]["role"] == "user":
            if isinstance(msgs[-1]["content"], str):
                msgs[-1]["content"] += "\n" + prompt_text + ctx_suffix
            else:
                msgs.append({"role": "user", "content": prompt_text + ctx_suffix})
        elif prompt_text:
            msgs.append({"role": "user", "content": prompt_text + ctx_suffix})
        return msgs

    base_model, base_provider, base_api_url = _resolve_model_provider(npc, team, model, provider)

    def _run_single(run_model, run_provider, run_npc, run_team, run_context, extra_kwargs):
        system_message = get_system_message(run_npc, run_team) if run_npc is not None else "You are a helpful assistant."
        ctx_suffix = _context_suffix(run_context)
        run_messages = _build_messages(messages, system_message, prompt, ctx_suffix)
        return get_litellm_response(
            (prompt + ctx_suffix) if prompt else None,
            messages=run_messages,
            model=run_model,
            provider=run_provider,
            api_url=extra_kwargs.get("api_url", base_api_url),
            api_key=api_key,
            images=images,
            attachments=attachments,
            stream=stream,
            include_usage=include_usage,
            **{k: v for k, v in extra_kwargs.items() if k != "api_url"},
        )

    use_matrix = matrix is not None and len(matrix) > 0
    multi_sample = n_samples and n_samples > 1

    if not use_matrix and not multi_sample:
        resolved_model, resolved_provider, resolved_api_url = _resolve_model_provider(npc, team, base_model, base_provider)
        return _run_single(resolved_model, resolved_provider, npc, team, context, {"api_url": resolved_api_url, **kwargs})

    combos = []
    if use_matrix:
        keys = list(matrix.keys())
        values = []
        for key in keys:
            val = matrix[key]
            if isinstance(val, (list, tuple, set)):
                values.append(list(val))
            else:
                values.append([val])
        for combo_values in itertools.product(*values):
            combo = dict(zip(keys, combo_values))
            combos.append(combo)
    else:
        combos.append({})

    runs = []
    for combo in combos:
        run_model = combo.get("model", base_model)
        run_provider = combo.get("provider", base_provider)
        run_npc = combo.get("npc", npc)
        run_team = combo.get("team", team)
        run_context = combo.get("context", context)
        extra_kwargs = dict(kwargs)
        for k, v in combo.items():
            if k not in {"model", "provider", "npc", "context", "team"}:
                extra_kwargs[k] = v

        resolved_model, resolved_provider, resolved_api_url = _resolve_model_provider(run_npc, run_team, run_model, run_provider)
        for sample_idx in range(max(1, n_samples or 1)):
            resp = _run_single(resolved_model, resolved_provider, run_npc, run_team, run_context, {"api_url": resolved_api_url, **extra_kwargs})
            runs.append({
                "response": resp.get("response") if isinstance(resp, dict) else resp,
                "raw": resp,
                "combo": combo,
                "sample_index": sample_idx,
            })

    aggregated = {
        "response": runs[0]["response"] if runs else None,
        "runs": runs,
    }
    if runs and isinstance(runs[0]["raw"], dict) and "messages" in runs[0]["raw"]:
        aggregated["messages"] = runs[0]["raw"].get("messages")
    return aggregated




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





def _get_jinxs(npc, team):
    """Get available jinxs from npc and team."""
    jinxs = {}
    if npc and hasattr(npc, 'jinxs_dict'):
        jinxs.update(npc.jinxs_dict)
    if team and hasattr(team, 'jinxs_dict'):
        jinxs.update(team.jinxs_dict)
    return jinxs


def _jinxs_to_tools(jinxs):
    """Convert jinxs to OpenAI-style tool definitions."""
    return [jinx.to_tool_def() for jinx in jinxs.values()]


def _execute_jinx(jinx, inputs, npc, team, messages, extra_globals):
    """Execute a jinx and return output."""
    try:
        jinja_env = None
        if npc and hasattr(npc, 'jinja_env'):
            jinja_env = npc.jinja_env
        elif team and hasattr(team, 'forenpc') and team.forenpc:
            jinja_env = getattr(team.forenpc, 'jinja_env', None)

        # Merge with default values from jinx definition
        full_inputs = {}
        for inp in getattr(jinx, 'inputs', []):
            if isinstance(inp, dict):
                key = list(inp.keys())[0]
                default_val = inp[key]
                # Expand paths
                if isinstance(default_val, str) and default_val.startswith("~"):
                    default_val = os.path.expanduser(default_val)
                full_inputs[key] = default_val
            else:
                full_inputs[inp] = ""
        # Override with provided inputs
        full_inputs.update(inputs or {})

        result = jinx.execute(
            input_values=full_inputs,
            npc=npc,
            messages=messages,
            extra_globals=extra_globals,
            jinja_env=jinja_env
        )
        if result is None:
            return "Executed with no output."
        if not isinstance(result, dict):
            return str(result)
        return result.get("output", str(result))
    except Exception as e:
        return f"Error: {e}"


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
    actions: Dict[str, Dict] = None,  # backwards compat, ignored
    extra_globals=None,
    max_iterations: int = 5,
    jinxs: Dict = None,
    tool_capable: bool = None,  # If None, will be auto-detected
):
    """
    Simple agent loop: try tool calling first, fall back to ReAct if unsupported.
    """
    if messages is None:
        messages = []

    total_usage = {"input_tokens": 0, "output_tokens": 0}

    # Use provided jinxs or get from npc/team
    if jinxs is None:
        jinxs = _get_jinxs(npc, team)

    # Keep full message history
    full_messages = messages.copy() if messages else []

    # If we have jinxs, use ReAct fallback (JSON prompting) instead of tool_calls
    if jinxs:
        return _react_fallback(
            command, 
            model, 
            provider, 
            api_url, 
            api_key, 
            npc, 
            team,
            full_messages, 
            images, 
            stream, 
            context, 
            jinxs, 
            extra_globals, 
            max_iterations
        )

    # No jinxs - just get a direct response
    try:
        response = get_llm_response(
            command,
            model=model,
            provider=provider,
            api_url=api_url,
            api_key=api_key,
            messages=messages[-10:],
            npc=npc,
            team=team,
            images=images,
            stream=stream,
            context=context,
        )
    except Exception as e:
        print(f"[check_llm_command] EXCEPTION in get_llm_response: {type(e).__name__}: {e}", "red")
        return {
            "messages": full_messages,
            "output": f"LLM call failed: {e}",
            "error": str(e),
            "usage": total_usage,
        }

    if response.get("usage"):
        total_usage["input_tokens"] += response["usage"].get("input_tokens", 0)
        total_usage["output_tokens"] += response["usage"].get("output_tokens", 0)

    full_messages.append({"role": "user", "content": command})
    assistant_response = response.get("response", "")
    if assistant_response and isinstance(assistant_response, str):
        full_messages.append({"role": "assistant", "content": assistant_response})
    return {
        "messages": full_messages,
        "output": assistant_response,
        "usage": total_usage,
    }


def _react_fallback(
    command, 
    model, 
    provider, 
    api_url, 
    api_key, 
    npc, 
    team,
    messages, 
    images, 
    stream, 
    context, 
    jinxs, 
    extra_globals, 
    max_iterations
):
    """ReAct-style fallback for models without tool calling."""
    import logging
    logger = logging.getLogger("npcpy.llm_funcs")
    logger.debug(f"[_react_fallback] Starting with {len(messages) if messages else 0} messages, jinxs: {list(jinxs.keys()) if jinxs else None}")

    total_usage = {"input_tokens": 0, "output_tokens": 0}
    current_messages = messages.copy() if messages else []
    jinx_executions = []  # Track jinx calls for UI display
    generated_images = []  # Track images generated by jinxs for subsequent LLM calls
    logger.debug(f"[_react_fallback] current_messages initialized with {len(current_messages)} messages")

    # Build jinx list with input parameters
    def _jinx_info(name, jinx):
        inputs = getattr(jinx, 'inputs', [])
        input_names = [i if isinstance(i, str) else list(i.keys())[0] for i in inputs]
        return f"- {name}({', '.join(input_names)}): {jinx.description}"

    jinx_list = "\n".join(_jinx_info(n, j) for n, j in jinxs.items()) if jinxs else "None"

    # Cap iterations - after this, return to orchestrator for review/compression
    effective_max = min(max_iterations, 7)

    for iteration in range(effective_max):
        # Build history of what's been tried
        history_text = ""
        if jinx_executions:
            history_text = "\n\nPrevious tool calls this session:\n" + "\n".join(
                f"- {h['name']}({h['inputs']}) -> {h['output']}"
                for h in jinx_executions[-5:]
            )

        prompt = f"""Request: {command}

Available Tools:
{jinx_list}

Instructions:
1. Analyze the request and determine the best tool to use
2. If you have enough information to answer, use {{"action": "answer", "response": "your answer"}}
3. If you need to use a tool, use {{"action": "jinx", "jinx_name": "tool_name", "inputs": {{"param": "value"}}}}
4. Use EXACT parameter names from tool definitions
5. Do NOT repeat the same tool call with the same inputs{history_text}"""

        if context:
            prompt += f"\n\nCurrent context: {context}"

        response = get_llm_response(
            prompt,
            model=model,
            provider=provider,
            api_url=api_url,
            api_key=api_key,
            messages=current_messages[-10:],
            npc=npc,
            team=team,
            images=((images or []) if iteration == 0 else []) + generated_images or None,
            format="json",
            context=context,
        )

        if response.get("usage"):
            total_usage["input_tokens"] += response["usage"].get("input_tokens", 0)
            total_usage["output_tokens"] += response["usage"].get("output_tokens", 0)

        decision = response.get("response", {})
        logger.debug(f"[_react_fallback] Raw decision: {str(decision)[:200]}")
        print(f"[REACT-DEBUG] Full response keys: {response.keys()}")
        print(f"[REACT-DEBUG] Raw response['response']: {str(response.get('response', 'NONE'))[:500]}")
        print(f"[REACT-DEBUG] Raw decision type: {type(decision)}, value: {str(decision)[:500]}")

        # Handle None response - model decided no action needed
        if decision is None:
            logger.debug(f"[_react_fallback] Decision is None, returning current output")
            return {"messages": current_messages, "output": "", "usage": total_usage, "jinx_executions": jinx_executions}

        if isinstance(decision, str):
            try:
                decision = json.loads(decision)
            except:
                logger.debug(f"[_react_fallback] Could not parse JSON, returning as text")
                return {"messages": current_messages, "output": decision, "usage": total_usage, "jinx_executions": jinx_executions}

        logger.debug(f"[_react_fallback] Parsed decision action: {decision.get('action') if decision else 'None'}")
        if decision.get("action") == "answer":
            output = decision.get("response", "")

            # Prepend any image URLs from jinx executions to the output
            import re
            for jexec in jinx_executions:
                joutput = jexec.get("output", "")
                if joutput:
                    # Find image URLs in jinx output
                    img_urls = re.findall(r'/uploads/[^\s,\'"]+\.(?:png|jpg|jpeg|webp|gif)', str(joutput), re.IGNORECASE)
                    for url in img_urls:
                        if url not in output:
                            output = f"![Generated Image]({url})\n\n{output}"

            # Add user message to full history
            current_messages.append({"role": "user", "content": command})
            if stream:
                # Use truncated for API, keep full history
                final = get_llm_response(command, model=model, provider=provider, messages=current_messages[-10:], npc=npc, team=team, stream=True, context=context)
                if final.get("usage"):
                    total_usage["input_tokens"] += final["usage"].get("input_tokens", 0)
                    total_usage["output_tokens"] += final["usage"].get("output_tokens", 0)
                # Return full messages, not truncated from final
                logger.debug(f"[_react_fallback] Answer (stream) - returning {len(current_messages)} messages")
                return {"messages": current_messages, "output": final.get("response", output), "usage": total_usage, "jinx_executions": jinx_executions}
            # Non-streaming: add assistant response to full history
            if output and isinstance(output, str):
                current_messages.append({"role": "assistant", "content": output})
            logger.debug(f"[_react_fallback] Answer - returning {len(current_messages)} messages")
            return {"messages": current_messages, "output": output, "usage": total_usage, "jinx_executions": jinx_executions}

        elif decision.get("action") == "jinx" or decision.get("action") in jinxs:
            # Handle multiple formats:
            # 1. {"action": "jinx", "jinx_name": "foo", "inputs": {...}}
            # 2. {"action": "foo", "inputs": {...}}
            # 3. {"action": "foo", "param1": "...", "param2": "..."} - params at top level
            jinx_name = decision.get("jinx_name") or decision.get("action")
            inputs = decision.get("inputs", {})

            # If inputs is empty, check if params are at top level
            if not inputs:
                # Extract all keys except 'action', 'jinx_name', 'inputs' as potential inputs
                inputs = {k: v for k, v in decision.items() if k not in ('action', 'jinx_name', 'inputs', 'response')}
            logger.debug(f"[_react_fallback] Jinx action: {jinx_name} with inputs: {inputs}")
            print(f"[REACT-DEBUG] Chose jinx: {jinx_name}, inputs: {str(inputs)[:200]}")

            if jinx_name not in jinxs:
                context = f"Error: '{jinx_name}' not found. Available: {list(jinxs.keys())}"
                logger.debug(f"[_react_fallback] Jinx not found: {jinx_name}")
                continue

            # Validate required parameters before executing
            jinx_obj = jinxs[jinx_name]
            # Handle both dict and Jinx object
            if hasattr(jinx_obj, 'inputs'):
                required_inputs = jinx_obj.inputs or []
            elif isinstance(jinx_obj, dict):
                required_inputs = jinx_obj.get('inputs', [])
            else:
                required_inputs = []

            if required_inputs:
                # Get just the parameter names (handle both string and dict formats)
                # String inputs are required, dict inputs have defaults and are optional
                required_names = []
                optional_names = []
                for inp in required_inputs:
                    if isinstance(inp, str):
                        # String inputs have no default, so they're required
                        required_names.append(inp)
                    elif isinstance(inp, dict):
                        # Dict inputs have default values, so they're optional
                        optional_names.extend(inp.keys())

                # Check which required params are missing (only string inputs, not dict inputs with defaults)
                missing = [p for p in required_names if p not in inputs or not inputs.get(p)]
                provided = list(inputs.keys())
                if missing:
                    all_params = required_names + optional_names
                    context = f"Error: jinx '{jinx_name}' requires parameters {required_names} but got {provided}. Missing: {missing}. Optional params with defaults: {optional_names}. Please retry with correct parameter names."
                    logger.debug(f"[_react_fallback] Missing required params: {missing}")
                    print(f"[REACT-DEBUG] Missing params for {jinx_name}: {missing}, got: {provided}")
                    continue

            logger.debug(f"[_react_fallback] Executing jinx: {jinx_name}")
            output = _execute_jinx(jinxs[jinx_name], inputs, npc, team, current_messages, extra_globals)
            logger.debug(f"[_react_fallback] Jinx output: {str(output)[:200]}")
            jinx_executions.append({
                "name": jinx_name,
                "inputs": inputs,
                "output": str(output) if output else None
            })

            # Extract generated image paths from output for subsequent LLM calls
            import re
            image_paths = re.findall(r'/uploads/[^\s,\'"]+\.(?:png|jpg|jpeg|webp|gif)', str(output), re.IGNORECASE)
            if image_paths:
                # Convert relative URLs to absolute file paths
                for img_path in image_paths:
                    # Remove leading slash
                    local_path = img_path.lstrip('/')
                    # Check various possible locations
                    if os.path.exists(local_path):
                        generated_images.append(local_path)
                        print(f"[REACT-DEBUG] Added generated image: {local_path}")
                    elif os.path.exists(os.path.join(os.getcwd(), local_path)):
                        full_path = os.path.join(os.getcwd(), local_path)
                        generated_images.append(full_path)
                        print(f"[REACT-DEBUG] Added generated image (cwd): {full_path}")
                    else:
                        # Just add the URL path anyway - let get_llm_response handle it
                        generated_images.append(local_path)
                        print(f"[REACT-DEBUG] Added generated image (not found, using anyway): {local_path}")

            # Truncate output for context to avoid sending huge base64 data back to LLM
            output_for_context = str(output)[:8000] + "..." if len(str(output)) > 8000 else str(output)
            context = f"Tool '{jinx_name}' returned: {output_for_context}"
            command = f"{command}\n\nPrevious: {context}"

        else:
            logger.debug(f"[_react_fallback] Unknown action - returning {len(current_messages)} messages")
            # If we have jinx executions, return the last output instead of empty decision
            if jinx_executions and jinx_executions[-1].get("output"):
                return {"messages": current_messages, "output": jinx_executions[-1]["output"], "usage": total_usage, "jinx_executions": jinx_executions}
            # If decision is empty {}, retry with clearer prompt if jinxs are available
            if not decision or decision == {}:
                if jinxs and iteration < max_iterations - 1:
                    # Retry with explicit instruction to use a jinx
                    print(f"[REACT-DEBUG] Empty decision on iteration {iteration}, retrying with clearer prompt")
                    context = f"You MUST use one of these tools to complete the task: {list(jinxs.keys())}. Return JSON with action and inputs."
                    continue
                else:
                    # Last resort: get a text response
                    print(f"[REACT-DEBUG] Empty decision, getting text response instead")
                    current_messages.append({"role": "user", "content": command})
                    fallback_response = get_llm_response(
                        command,
                        model=model,
                        provider=provider,
                        messages=current_messages[-10:],
                        npc=npc,
                        team=team,
                        stream=stream,
                        context=context,
                    )
                    if fallback_response.get("usage"):
                        total_usage["input_tokens"] += fallback_response["usage"].get("input_tokens", 0)
                        total_usage["output_tokens"] += fallback_response["usage"].get("output_tokens", 0)
                    output = fallback_response.get("response", "")
                    if output and isinstance(output, str):
                        current_messages.append({"role": "assistant", "content": output})
                    return {"messages": current_messages, "output": output, "usage": total_usage, "jinx_executions": jinx_executions}
            return {"messages": current_messages, "output": str(decision), "usage": total_usage, "jinx_executions": jinx_executions}

    logger.debug(f"[_react_fallback] Max iterations - returning {len(current_messages)} messages")
    # If we have jinx executions, return the last output
    if jinx_executions and jinx_executions[-1].get("output"):
        return {"messages": current_messages, "output": jinx_executions[-1]["output"], "usage": total_usage, "jinx_executions": jinx_executions}
    return {"messages": current_messages, "output": f"Max iterations reached. Last: {context}", "usage": total_usage, "jinx_executions": jinx_executions}






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
                       n_tries=n_attempts, # Corrected from n_tries to n_attempts
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
    if attempt_number <= n_attempts: # Corrected logic: retry if attempt_number is within limits
        if not response["response"].get("related_facts", []): # Only retry if no related facts found
            print(f"  Attempt {attempt_number} to find related facts yielded no results. Retrying...")
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
