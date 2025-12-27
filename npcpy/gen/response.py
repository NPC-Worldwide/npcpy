from typing import Any, Dict, List, Union
from pydantic import BaseModel
from npcpy.data.image import compress_image
from npcpy.npc_sysenv import get_system_message, lookup_provider, render_markdown
import base64
import json
import yaml
import uuid
import os
import logging

logger = logging.getLogger(__name__)

try:
    import ollama
except ImportError:
    pass
except OSError:
    logger.warning("Ollama is not installed or not available.")

try:
    import litellm
    from litellm import completion
except ImportError:
    pass
except OSError:
    pass

# Token costs per 1M tokens (input, output)
TOKEN_COSTS = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3": (10.00, 40.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # Anthropic
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
    # Google
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-3-pro": (2.00, 12.00),
    # Groq (free tier limits, paid is cheap)
    "llama-3": (0.05, 0.08),
    "llama-3.1": (0.05, 0.08),
    "llama-3.2": (0.05, 0.08),
    "llama-4": (0.05, 0.10),
    "mixtral": (0.24, 0.24),
    # DeepSeek
    "deepseek-v3": (0.27, 1.10),
    "deepseek-r1": (0.55, 2.19),
    # Mistral
    "mistral-large": (2.00, 6.00),
    "mistral-small": (0.20, 0.60),
    # xAI
    "grok-2": (2.00, 10.00),
    "grok-3": (3.00, 15.00),
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a response."""
    if not model:
        return 0.0

    # Normalize model name - strip provider prefix and lowercase
    model_key = model.split("/")[-1].lower()

    # Check for exact or partial match
    costs = None
    for key, cost in TOKEN_COSTS.items():
        if key in model_key or model_key in key:
            costs = cost
            break

    if not costs:
        return 0.0  # Unknown/local model, assume free

    input_cost, output_cost = costs
    return (input_tokens * input_cost / 1_000_000) + (output_tokens * output_cost / 1_000_000)

def handle_streaming_json(api_params):
    """
    Handles streaming responses when JSON format is requested from LiteLLM.
    """
    json_buffer = ""
    stream = completion(**api_params)
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            json_buffer += content
            try:
                json.loads(json_buffer)
                yield chunk
            except json.JSONDecodeError:
                pass

def get_transformers_response(
   prompt: str = None,
   model=None,
   tokenizer=None, 
   tools: list = None,
   tool_map: Dict = None,
   format: str = None,
   messages: List[Dict[str, str]] = None,
   auto_process_tool_calls: bool = False,
   **kwargs,
) -> Dict[str, Any]:
   import torch
   import json
   import uuid
   from transformers import AutoTokenizer, AutoModelForCausalLM
   
   result = {
       "response": None,
       "messages": messages.copy() if messages else [],
       "raw_response": None,
       "tool_calls": [], 
       "tool_results": []
   }
   
   if model is None or tokenizer is None:
       model_name = model if isinstance(model, str) else "Qwen/Qwen3-1.7b"
       tokenizer = AutoTokenizer.from_pretrained(model_name)
       model = AutoModelForCausalLM.from_pretrained(model_name)
       
       if tokenizer.pad_token is None:
           tokenizer.pad_token = tokenizer.eos_token
   
   if prompt:
       if result['messages'] and result['messages'][-1]["role"] == "user":
           result['messages'][-1]["content"] = prompt
       else:
           result['messages'].append({"role": "user", "content": prompt})
   
   if format == "json":
       json_instruction = """If you are returning a json object, begin directly with the opening {.
Do not include any additional markdown formatting or leading ```json tags in your response."""
       if result["messages"] and result["messages"][-1]["role"] == "user":
           result["messages"][-1]["content"] += "\n" + json_instruction

   chat_text = tokenizer.apply_chat_template(result["messages"], tokenize=False, add_generation_prompt=True)
   device = next(model.parameters()).device
   inputs = tokenizer(chat_text, return_tensors="pt", padding=True, truncation=True)
   inputs = {k: v.to(device) for k, v in inputs.items()}
   
       
   with torch.no_grad():
       outputs = model.generate(
           **inputs,
           max_new_tokens=256,
           temperature=0.7,
           do_sample=True,
           pad_token_id=tokenizer.eos_token_id,
       )
   
   response_content = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
   result["response"] = response_content
   result["raw_response"] = response_content
   result["messages"].append({"role": "assistant", "content": response_content})

   if auto_process_tool_calls and tools and tool_map:
       detected_tools = []
       for tool in tools:
           tool_name = tool.get("function", {}).get("name", "")
           if tool_name in response_content:
               detected_tools.append({
                   "id": str(uuid.uuid4()),
                   "function": {
                       "name": tool_name,
                       "arguments": "{}"
                   }
               })
       
       if detected_tools:
           result["tool_calls"] = detected_tools
           result = process_tool_calls(result, tool_map, "local", "transformers", result["messages"], tools=tools)
   
   if format == "json":
       try:
           if response_content.startswith("```json"):
               response_content = response_content.replace("```json", "").replace("```", "").strip()
           parsed_response = json.loads(response_content)
           result["response"] = parsed_response
       except json.JSONDecodeError:
           result["error"] = f"Invalid JSON response: {response_content}"
   
   return result

        
def get_ollama_response(
    prompt: str,
    model: str,
    images: List[str] = None,
    tools: list = None,
    tool_choice: Dict = None,
    tool_map: Dict = None,
    think= None ,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    stream: bool = False,
    attachments: List[str] = None,
    auto_process_tool_calls: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generates a response using the Ollama API, supporting both streaming and non-streaming.
    """

    options = {}

    image_paths = []
    if images:
        image_paths.extend(images)
    
    if attachments:
        for attachment in attachments:
            if os.path.exists(attachment):
                _, ext = os.path.splitext(attachment)
                ext = ext.lower()
                
                if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    image_paths.append(attachment)
                elif ext == '.pdf':
                    try:
                        from npcpy.data.load import load_pdf
                        pdf_data = load_pdf(attachment)
                        if pdf_data is not None:
                            if prompt:
                                prompt += f"\n\nContent from PDF: {os.path.basename(attachment)}\n{pdf_data[:5000]}..."
                            else:
                                prompt = f"Content from PDF: {os.path.basename(attachment)}\n{pdf_data[:5000]}..."
                    except Exception:
                        pass
                elif ext == '.csv':
                    try:
                        from npcpy.data.load import load_csv
                        csv_data = load_csv(attachment)
                        if csv_data is not None:
                            csv_sample = csv_data.head(100).to_string()
                            if prompt:
                                prompt += f"\n\nContent from CSV: {os.path.basename(attachment)} (first 100 rows):\n{csv_sample} \n csv description: {csv_data.describe()}"
                            else:
                                prompt = f"Content from CSV: {os.path.basename(attachment)} (first 100 rows):\n{csv_sample} \n csv description: {csv_data.describe()}"
                    except Exception:
                        pass


    if prompt:
        if messages and messages[-1]["role"] == "user":
            if isinstance(messages[-1]["content"], str):
                messages[-1]["content"] = prompt
            elif isinstance(messages[-1]["content"], list):
                for i, item in enumerate(messages[-1]["content"]):
                    if item.get("type") == "text":
                        messages[-1]["content"][i]["text"] = prompt
                        break
                else:
                    messages[-1]["content"].append({"type": "text", "text": prompt})
        else:
            if not messages:
                messages = []
            messages.append({"role": "user", "content": prompt})
    if format == "json" and not stream:
        json_instruction = """If you are a returning a json object, begin directly with the opening {.
            If you are returning a json array, begin directly with the opening [.
            Do not include any additional markdown formatting or leading
            ```json tags in your response. The item keys should be based on the ones provided
            by the user. Do not invent new ones."""

        if messages and messages[-1]["role"] == "user":
            if isinstance(messages[-1]["content"], list):
                messages[-1]["content"].append({
                    "type": "text",
                    "text": json_instruction
                })
            elif isinstance(messages[-1]["content"], str):
                messages[-1]["content"] += "\n" + json_instruction

    if format == "yaml" and not stream:
        yaml_instruction = """Return your response as valid YAML. Do not include ```yaml markdown tags.
            For multi-line strings like code, use the literal block scalar (|) syntax:
            code: |
              your code here
              more lines here
            The keys should be based on the ones requested by the user. Do not invent new ones."""

        if messages and messages[-1]["role"] == "user":
            if isinstance(messages[-1]["content"], list):
                messages[-1]["content"].append({
                    "type": "text",
                    "text": yaml_instruction
                })
            elif isinstance(messages[-1]["content"], str):
                messages[-1]["content"] += "\n" + yaml_instruction

    if image_paths:
        last_user_idx = -1
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                last_user_idx = i
        if last_user_idx == -1:
            messages.append({"role": "user", "content": ""})
            last_user_idx = len(messages) - 1
        messages[last_user_idx]["images"] = image_paths

    # Ollama's pydantic model requires tool_calls arguments to be dicts, not strings
    for msg in messages:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("function") and isinstance(tc["function"].get("arguments"), str):
                    try:
                        tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        tc["function"]["arguments"] = {}

    api_params = {
        "model": model,
        "messages": messages,
        "stream": stream if not (tools and tool_map and auto_process_tool_calls) else False,
    }

    if tools:
        api_params["tools"] = tools
        if tool_choice:
            options["tool_choice"] = tool_choice


    if think is not None:
        api_params['think'] = think

    if isinstance(format, type) and not stream:
        api_params["format"] = format.model_json_schema()
    elif isinstance(format, str) and format == "json" and not stream:
        api_params["format"] = "json"

    for key, value in kwargs.items():
        if key in [
            "stop", 
            "temperature", 
            "top_p", 
            "max_tokens",
            "max_completion_tokens",
            "extra_headers", 
            "parallel_tool_calls",
            "response_format",
            "user",
        ]:
            options[key] = value

    result = {
        "response": None,
        "messages": messages.copy(),
        "raw_response": None,
        "tool_calls": [], 
        "tool_results": []
    }

    

    
    if not auto_process_tool_calls or not (tools and tool_map):
        res = ollama.chat(**api_params, options=options)
        result["raw_response"] = res

        if stream:
            # Return immediately for streaming - don't check 'in' on generator as it consumes it
            result["response"] = res
            return result

        # Extract usage from ollama response (only for non-streaming)
        if hasattr(res, 'prompt_eval_count') or 'prompt_eval_count' in res:
            input_tokens = getattr(res, 'prompt_eval_count', None) or res.get('prompt_eval_count', 0) or 0
            output_tokens = getattr(res, 'eval_count', None) or res.get('eval_count', 0) or 0
            result["usage"] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        else:

            message = res.get("message", {})
            response_content = message.get("content", "")
            result["response"] = response_content
            result["messages"].append({"role": "assistant", "content": response_content})
            
            if message.get('tool_calls'):
                result["tool_calls"] = message['tool_calls']
            
            
            if format == "json":
                try:
                    if isinstance(response_content, str):
                        if response_content.startswith("```json"):
                            response_content = (
                                response_content.replace("```json", "")
                                .replace("```", "")
                                .strip()
                            )
                        parsed_response = json.loads(response_content)
                        result["response"] = parsed_response
                except json.JSONDecodeError:
                    result["error"] = f"Invalid JSON response: {response_content}"

            if format == "yaml":
                try:
                    if isinstance(response_content, str):
                        if response_content.startswith("```yaml"):
                            response_content = (
                                response_content.replace("```yaml", "")
                                .replace("```", "")
                                .strip()
                            )
                        parsed_response = yaml.safe_load(response_content)
                        result["response"] = parsed_response
                except yaml.YAMLError:
                    result["error"] = f"Invalid YAML response: {response_content}"

            return result


    logger.debug(f"ollama api_params: {api_params}")
    res = ollama.chat(**api_params, options=options)
    result["raw_response"] = res
    
    
    
    message = res.get("message", {})
    response_content = message.get("content", "")
    
    
    if message.get('tool_calls'):

        
        result["tool_calls"] = message['tool_calls']
        
        response_for_processing = {
            "response": response_content,
            "raw_response": res,
            "messages": messages,
            "tool_calls": message['tool_calls']
        }
        
        
        processed_result = process_tool_calls(response_for_processing,
                                              tool_map, model,
                                              'ollama',
                                              messages,
                                              stream=False,
                                              tools=tools)
        
        
        if stream:

            
            
            final_messages = processed_result["messages"]
            
            
            final_api_params = {
                "model": model,
                "messages": final_messages,
                "stream": True,
            }
            
            if tools:
                final_api_params["tools"] = tools
            
            final_stream = ollama.chat(**final_api_params, options=options)
            processed_result["response"] = final_stream
            
        return processed_result
    
    
    else:
        result["response"] = response_content
        result["messages"].append({"role": "assistant", "content": response_content})
        
        if stream:
            
            stream_api_params = {
                "model": model,
                "messages": messages,
                "stream": True,
            }
            if tools:
                stream_api_params["tools"] = tools
            
            result["response"] = ollama.chat(**stream_api_params, options=options)
        else:

            if format == "json":
                try:
                    llm_response = response_content
                    if isinstance(llm_response, str):
                        llm_response = llm_response.strip()
                        
                        if '```json' in llm_response:
                            start = llm_response.find('```json') + 7
                            end = llm_response.rfind('```')
                            if end > start:
                                llm_response = llm_response[start:end].strip()
                        
                        first_brace = llm_response.find('{')
                        first_bracket = llm_response.find('[')
                        
                        if first_brace == -1 and first_bracket == -1:
                            result["response"] = {}
                            result["error"] = "No JSON found in response"
                            return result
                        
                        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
                            llm_response = llm_response[first_brace:]
                            last_brace = llm_response.rfind('}')
                            if last_brace != -1:
                                llm_response = llm_response[:last_brace+1]
                        else:
                            llm_response = llm_response[first_bracket:]
                            last_bracket = llm_response.rfind(']')
                            if last_bracket != -1:
                                llm_response = llm_response[:last_bracket+1]
                        
                        parsed_json = json.loads(llm_response, strict=False)
                        
                        if "json" in parsed_json:
                            result["response"] = parsed_json["json"]
                        else:
                            result["response"] = parsed_json
                        
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug(f"JSON parsing error: {str(e)}, raw response: {llm_response[:500]}")
                    result["response"] = {}
                    result["error"] = "Invalid JSON response"

            if format == "yaml":
                try:
                    if isinstance(llm_response, str):
                        llm_response = llm_response.strip()

                        if '```yaml' in llm_response:
                            start = llm_response.find('```yaml') + 7
                            end = llm_response.rfind('```')
                            if end > start:
                                llm_response = llm_response[start:end].strip()

                        parsed_yaml = yaml.safe_load(llm_response)
                        result["response"] = parsed_yaml

                except (yaml.YAMLError, TypeError) as e:
                    logger.debug(f"YAML parsing error: {str(e)}, raw response: {llm_response[:500]}")
                    result["response"] = {}
                    result["error"] = "Invalid YAML response"

        return result

import time


def get_llamacpp_response(
    prompt: str = None,
    model: str = None,
    images: List[str] = None,
    tools: list = None,
    tool_choice: Dict = None,
    tool_map: Dict = None,
    think=None,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    stream: bool = False,
    attachments: List[str] = None,
    auto_process_tool_calls: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate response using llama-cpp-python for local GGUF/GGML files.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        return {
            "response": "",
            "messages": messages or [],
            "error": "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
        }

    result = {
        "response": None,
        "messages": messages.copy() if messages else [],
        "raw_response": None,
        "tool_calls": [],
        "tool_results": []
    }

    if prompt:
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] = prompt
        else:
            if not messages:
                messages = []
            messages.append({"role": "user", "content": prompt})

    try:
        # Load model
        n_ctx = kwargs.get("n_ctx", 4096)
        n_gpu_layers = kwargs.get("n_gpu_layers", -1)  # -1 = all layers on GPU if available

        llm = Llama(
            model_path=model,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )

        # Build params
        params = {
            "messages": messages,
            "stream": stream,
        }
        if kwargs.get("temperature"):
            params["temperature"] = kwargs["temperature"]
        if kwargs.get("max_tokens"):
            params["max_tokens"] = kwargs["max_tokens"]
        if kwargs.get("top_p"):
            params["top_p"] = kwargs["top_p"]
        if kwargs.get("stop"):
            params["stop"] = kwargs["stop"]

        if stream:
            response = llm.create_chat_completion(**params)

            def generate():
                for chunk in response:
                    # Yield the full chunk dict for proper streaming handling
                    yield chunk

            result["response"] = generate()
        else:
            response = llm.create_chat_completion(**params)
            result["raw_response"] = response

            if response.get("choices"):
                content = response["choices"][0].get("message", {}).get("content", "")
                result["response"] = content
                result["messages"].append({"role": "assistant", "content": content})

            if response.get("usage"):
                result["usage"] = {
                    "input_tokens": response["usage"].get("prompt_tokens", 0),
                    "output_tokens": response["usage"].get("completion_tokens", 0),
                }

    except Exception as e:
        result["error"] = f"llama.cpp error: {str(e)}"
        result["response"] = ""

    return result


def get_litellm_response(
    prompt: str = None,
    model: str = None,
    provider: str = None,
    images: List[str] = None,
    tools: list = None,
    tool_choice: Dict = None,
    tool_map: Dict = None,
    think= None,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    api_key: str = None,
    api_url: str = None,
    stream: bool = False,
    attachments: List[str] = None,
    auto_process_tool_calls: bool = False, 
    include_usage: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    result = {
        "response": None,
        "messages": messages.copy() if messages else [],
        "raw_response": None,
        "tool_calls": [], 
        "tool_results":[],
    }
    if provider == "ollama" and 'gpt-oss' not in model:
        return get_ollama_response(
            prompt, 
            model, 
            images=images, 
            tools=tools, 
            tool_choice=tool_choice, 
            tool_map=tool_map,
            think=think,
            format=format, 
            messages=messages, 
            stream=stream, 
            attachments=attachments, 
            auto_process_tool_calls=auto_process_tool_calls, 
            **kwargs
        )
    elif provider=='transformers':
        return get_transformers_response(
            prompt,
            model,
            images=images,
            tools=tools,
            tool_choice=tool_choice,
            tool_map=tool_map,
            think=think,
            format=format,
            messages=messages,
            stream=stream,
            attachments=attachments,
            auto_process_tool_calls=auto_process_tool_calls,
            **kwargs

        )
    elif provider == 'llamacpp':
        return get_llamacpp_response(
            prompt,
            model,
            images=images,
            tools=tools,
            tool_choice=tool_choice,
            tool_map=tool_map,
            think=think,
            format=format,
            messages=messages,
            stream=stream,
            attachments=attachments,
            auto_process_tool_calls=auto_process_tool_calls,
            **kwargs
        )
    elif provider == 'lmstudio' or (model and '.lmstudio' in str(model)):
        # LM Studio uses OpenAI-compatible API on port 1234
        # Also detect models with .lmstudio in path (e.g., /home/user/.lmstudio/models/...)
        api_url = api_url or "http://127.0.0.1:1234/v1"
        provider = "openai"
        api_key = api_key or "lm-studio"  # LM Studio doesn't require real API key
        # Default timeout for local CPU inference (can be overridden via kwargs)
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 300  # 5 minutes for CPU inference with large prompts
    elif provider == 'llamacpp-server':
        # llama.cpp server uses OpenAI-compatible API on port 8080
        api_url = api_url or "http://127.0.0.1:8080/v1"
        provider = "openai"
        api_key = api_key or "llamacpp"  # llama.cpp server doesn't require real API key
        # Default timeout for local CPU inference (can be overridden via kwargs)
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 300  # 5 minutes for CPU inference with large prompts

    if attachments:
        for attachment in attachments:
            if os.path.exists(attachment):
                _, ext = os.path.splitext(attachment)
                ext = ext.lower()
                
                if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    if not images:
                        images = []
                    images.append(attachment)
                elif ext == '.pdf':
                    try:
                        from npcpy.data.load import load_pdf
                        pdf_data = load_pdf(attachment)
                        if pdf_data is not None:
                            if prompt:
                                prompt += f"\n\nContent from PDF: {os.path.basename(attachment)}\n{pdf_data}..."
                            else:
                                prompt = f"Content from PDF: {os.path.basename(attachment)}\n{pdf_data}..."

                    except Exception:
                        pass
                elif ext == '.csv':
                    try:
                        from npcpy.data.load import load_csv
                        csv_data = load_csv(attachment)
                        if csv_data is not None:
                            csv_sample = csv_data.head(10).to_string()
                            if prompt:
                                prompt += f"\n\nContent from CSV: {os.path.basename(attachment)} (first 10 rows):\n{csv_sample}"
                            else:
                                prompt = f"Content from CSV: {os.path.basename(attachment)} (first 10 rows):\n{csv_sample}"
                    except Exception:
                        pass

    if prompt:
        if result['messages'] and result['messages'][-1]["role"] == "user":
            if isinstance(messages[-1]["content"], str):
                result['messages'][-1]["content"] = prompt
            elif isinstance(result['messages'][-1]["content"], list):
                for i, item in enumerate(result['messages'][-1]["content"]):
                    if item.get("type") == "text":
                        result['messages'][-1]["content"][i]["text"] = prompt
                        break
                else:
                    result['messages'][-1]["content"].append({"type": "text", "text": prompt})
        else:
            result['messages'].append({"role": "user", "content": prompt})

    if format == "json" and not stream:
        json_instruction = """If you are a returning a json object, begin directly with the opening {.
            If you are returning a json array, begin directly with the opening [.
            Do not include any additional markdown formatting or leading
            ```json tags in your response. The item keys should be based on the ones provided
            by the user. Do not invent new ones."""

        if result["messages"] and result["messages"][-1]["role"] == "user":
            if isinstance(result["messages"][-1]["content"], list):
                result["messages"][-1]["content"].append({"type": "text", "text": json_instruction})
            elif isinstance(result["messages"][-1]["content"], str):
                result["messages"][-1]["content"] += "\n" + json_instruction

    if format == "yaml" and not stream:
        yaml_instruction = """Return your response as valid YAML. Do not include ```yaml markdown tags.
            For multi-line strings like code, use the literal block scalar (|) syntax:
            code: |
              your code here
              more lines here
            The keys should be based on the ones requested by the user. Do not invent new ones."""

        if result["messages"] and result["messages"][-1]["role"] == "user":
            if isinstance(result["messages"][-1]["content"], list):
                result["messages"][-1]["content"].append({"type": "text", "text": yaml_instruction})
            elif isinstance(result["messages"][-1]["content"], str):
                result["messages"][-1]["content"] += "\n" + yaml_instruction

    if images:
        last_user_idx = -1
        for i, msg in enumerate(result["messages"]):
            if msg["role"] == "user":
                last_user_idx = i
        if last_user_idx == -1:
            result["messages"].append({"role": "user", "content": []})
            last_user_idx = len(result["messages"]) - 1
        if isinstance(result["messages"][last_user_idx]["content"], str):
            
            result["messages"][last_user_idx]["content"] = [{"type": "text", 
                                                             "text": result["messages"][last_user_idx]["content"]
                                                             }]

        elif not isinstance(result["messages"][last_user_idx]["content"], list):
            result["messages"][last_user_idx]["content"] = []
        for image_path in images:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(compress_image(image_file.read())).decode("utf-8")
                result["messages"][last_user_idx]["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                )


    

    api_params = {"messages": result["messages"]}

    if include_usage:
      litellm.include_cost_in_streaming_usage = True
      api_params['stream_options'] = {"include_usage": True}

    if api_url is not None and ('openai-like' in provider or provider == "openai-like" or provider == "openai"):
        api_params["api_base"] = api_url
        provider = "openai"
    
    
    if provider =='enpisi' and api_url is None:
        api_params['api_base'] = 'https://api.enpisi.com'
        if api_key is None:
            api_key = os.environ.get('NPC_STUDIO_LICENSE_KEY')
            api_params['api_key'] = api_key
        if '-npc' in model: 
            model = model.split('-npc')[0]
        provider = "openai"

    if isinstance(format, type) and issubclass(format, BaseModel):
        api_params["response_format"] = format
    if model is None:
        model = os.environ.get("NPCSH_CHAT_MODEL", "llama3.2")
    if provider is None:
        provider = os.environ.get("NPCSH_CHAT_PROVIDER")

    # For OpenAI-compatible endpoints with api_base, always prefix with provider
    # LiteLLM needs this to know how to route the request
    # Also handle file paths (starting with /) which contain slashes but still need prefix
    if "api_base" in api_params and provider == "openai":
        api_params["model"] = f"openai/{model}"
    elif "/" not in model or model.startswith("/"):
        # No provider prefix in model, or model is a file path - add provider prefix
        api_params["model"] = f"{provider}/{model}"
    else:
        api_params["model"] = model
    if api_key is not None: 
        api_params["api_key"] = api_key
    if tools: 
        api_params["tools"] = tools
    if tool_choice: 
        api_params["tool_choice"] = tool_choice
    
    if kwargs:
        for key, value in kwargs.items():
            if key in [
                "stop", "temperature", "top_p", "max_tokens", "max_completion_tokens",
                 "extra_headers", "parallel_tool_calls",
                "response_format", "user", "timeout",
            ]:
                api_params[key] = value

    if not auto_process_tool_calls or not (tools and tool_map):
        api_params["stream"] = stream
        resp = completion(**api_params)
        result["raw_response"] = resp

        # Extract usage if available (handles both standard litellm and ollama formats)
        if hasattr(resp, 'usage') and resp.usage:
            result["usage"] = {
                "input_tokens": getattr(resp.usage, 'prompt_tokens', 0) or 0,
                "output_tokens": getattr(resp.usage, 'completion_tokens', 0) or 0,
            }
        elif hasattr(resp, 'prompt_eval_count'):
            # Ollama format
            result["usage"] = {
                "input_tokens": getattr(resp, 'prompt_eval_count', 0) or 0,
                "output_tokens": getattr(resp, 'eval_count', 0) or 0,
            }

        if stream:
            result["response"] = resp  
            return result
        else:
            
            llm_response = resp.choices[0].message.content
            result["response"] = llm_response
            result["messages"].append({"role": "assistant", 
                                       "content": llm_response})
            
            
            if hasattr(resp.choices[0].message, 'tool_calls') and resp.choices[0].message.tool_calls:
                result["tool_calls"] = resp.choices[0].message.tool_calls
            if format == "json":
                try:
                    if isinstance(llm_response, str):
                        llm_response = llm_response.strip()
                        
                        if '```json' in llm_response:
                            start = llm_response.find('```json') + 7
                            end = llm_response.rfind('```')
                            if end > start:
                                llm_response = llm_response[start:end].strip()
                        
                        first_brace = llm_response.find('{')
                        first_bracket = llm_response.find('[')
                        
                        if first_brace == -1 and first_bracket == -1:
                            result["response"] = {}
                            result["error"] = "No JSON found in response"
                            return result
                        
                        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
                            llm_response = llm_response[first_brace:]
                            last_brace = llm_response.rfind('}')
                            if last_brace != -1:
                                llm_response = llm_response[:last_brace+1]
                        else:
                            llm_response = llm_response[first_bracket:]
                            last_bracket = llm_response.rfind(']')
                            if last_bracket != -1:
                                llm_response = llm_response[:last_bracket+1]
                        
                        parsed_json = json.loads(llm_response, strict=False)
                        
                        if "json" in parsed_json:
                            result["response"] = parsed_json["json"]
                        else:
                            result["response"] = parsed_json
                        
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug(f"JSON parsing error: {str(e)}, raw response: {llm_response[:500]}")
                    result["response"] = {}
                    result["error"] = "Invalid JSON response"

            if format == "yaml":
                try:
                    if isinstance(llm_response, str):
                        llm_response = llm_response.strip()

                        # Strip ```yaml markdown if present
                        if '```yaml' in llm_response:
                            start = llm_response.find('```yaml') + 7
                            end = llm_response.rfind('```')
                            if end > start:
                                llm_response = llm_response[start:end].strip()
                        elif '```' in llm_response:
                            # Generic code block
                            start = llm_response.find('```') + 3
                            # Skip any language identifier on the same line
                            newline = llm_response.find('\n', start)
                            if newline != -1:
                                start = newline + 1
                            end = llm_response.rfind('```')
                            if end > start:
                                llm_response = llm_response[start:end].strip()

                        parsed_yaml = yaml.safe_load(llm_response)
                        result["response"] = parsed_yaml

                except (yaml.YAMLError, TypeError) as e:
                    logger.debug(f"YAML parsing error: {str(e)}, raw response: {llm_response[:500]}")
                    result["response"] = {}
                    result["error"] = "Invalid YAML response"

            return result

    
    
    initial_api_params = api_params.copy()
    initial_api_params["stream"] = False

    try:
        resp = completion(**initial_api_params)
    except Exception as e:
        logger.error(f"litellm completion() failed: {type(e).__name__}: {e}")
        result["error"] = str(e)
        result["response"] = f"LLM call failed: {e}"
        return result

    result["raw_response"] = resp

    # Extract usage if available
    if hasattr(resp, 'usage') and resp.usage:
        result["usage"] = {
            "input_tokens": getattr(resp.usage, 'prompt_tokens', 0) or 0,
            "output_tokens": getattr(resp.usage, 'completion_tokens', 0) or 0,
        }

    if not resp.choices:
        result["response"] = "No response from model"
        return result

    has_tool_calls = hasattr(resp.choices[0].message, 'tool_calls') and resp.choices[0].message.tool_calls
    
    if has_tool_calls:
        result["tool_calls"] = resp.choices[0].message.tool_calls

        processed_result = process_tool_calls(result,
                                              tool_map,
                                              model,
                                              provider,
                                              result["messages"],
                                              stream=False,
                                              tools=tools)

        # Always do a follow-up call to get a proper response after tool execution
        # Convert tool interactions to a clean format for the follow-up call
        clean_messages = []
        tool_results_summary = []

        for msg in processed_result["messages"]:
            role = msg.get('role', '')
            if role == 'assistant' and 'tool_calls' in msg:
                # Skip the tool_calls message - we'll summarize results instead
                continue
            elif role == 'tool':
                # Collect tool results for summary
                content = msg.get('content', '')
                # Truncate very long results
                if len(content) > 2000:
                    content = content[:2000] + "... (truncated)"
                tool_results_summary.append(content)
            else:
                clean_messages.append(msg)

        # Add tool results as an assistant message summarizing what was done
        if tool_results_summary:
            clean_messages.append({
                "role": "assistant",
                "content": "I executed the requested tools. Here are the results:\n\n" + "\n\n".join(tool_results_summary)
            })

        # Add instruction for the LLM to provide a helpful response
        clean_messages.append({
            "role": "user",
            "content": "Based on the tool results above, provide a brief summary of what happened. Do NOT output any code - the tool has already executed. Just describe the results concisely."
        })

        final_api_params = api_params.copy()
        final_api_params["messages"] = clean_messages
        final_api_params["stream"] = stream
        if "tools" in final_api_params:
            del final_api_params["tools"]
        if "tool_choice" in final_api_params:
            del final_api_params["tool_choice"]

        final_resp = completion(**final_api_params)

        if stream:
            processed_result["response"] = final_resp
        else:
            if final_resp.choices:
                final_content = final_resp.choices[0].message.content
                processed_result["response"] = final_content
                processed_result["messages"].append({"role": "assistant", "content": final_content})
            else:
                # No choices returned, use the tool results summary directly
                if tool_results_summary:
                    processed_result["response"] = "\n\n".join(tool_results_summary)
                else:
                    processed_result["response"] = "Tool executed successfully."

        return processed_result
        
        
    else:
        llm_response = resp.choices[0].message.content
        result["messages"].append({"role": "assistant", "content": llm_response})
        
        if stream:
            def string_chunk_generator():
                chunk_size = 1
                for i, char in enumerate(llm_response):
                    yield type('MockChunk', (), {
                        'id': f'mock-chunk-{i}',
                        'object': 'chat.completion.chunk',
                        'created': int(time.time()),
                        'model': model or 'unknown',
                        'choices': [type('Choice', (), {
                            'index': 0,
                            'delta': type('Delta', (), {
                                'content': char,
                                'role': 'assistant' if i == 0 else None
                            })(),
                            'finish_reason': 'stop' if i == len(llm_response) - 1 else None
                        })()]
                    })()
            
            result["response"] = string_chunk_generator()
        else:
            result["response"] = llm_response
    return result            
def process_tool_calls(response_dict, tool_map, model, provider, messages, stream=False, tools=None):
    result = response_dict.copy()
    result["tool_results"] = []

    if "messages" not in result:
        result["messages"] = messages if messages else []

    tool_calls = result.get("tool_calls", [])

    if not tool_calls:
        return result

    # First, add the assistant message with tool_calls (required by Gemini and other providers)
    # This must come BEFORE the tool results
    tool_calls_for_message = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            tool_calls_for_message.append(tc)
        else:
            # Convert object to dict format
            tool_calls_for_message.append({
                "id": getattr(tc, "id", str(uuid.uuid4())),
                "type": "function",
                "function": {
                    "name": getattr(tc.function, "name", "") if hasattr(tc, "function") else "",
                    "arguments": getattr(tc.function, "arguments", "{}") if hasattr(tc, "function") else "{}"
                }
            })

    result["messages"].append({
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls_for_message
    })

    for tool_call in tool_calls:
        tool_id = str(uuid.uuid4())
        tool_name = None
        arguments = {}
        

        if isinstance(tool_call, dict):
            tool_id = tool_call.get("id", str(uuid.uuid4()))
            tool_name = tool_call.get("function", {}).get("name")
            arguments_str = tool_call.get("function", {}).get("arguments", "{}")
        else:
            tool_id = getattr(tool_call, "id", str(uuid.uuid4()))
            if hasattr(tool_call, "function"):
                func_obj = tool_call.function
                tool_name = getattr(func_obj, "name", None)
                arguments_str = getattr(func_obj, "arguments", "{}")
            else:
                continue

        try:
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        except json.JSONDecodeError:
            arguments = {"raw_arguments": arguments_str}
        
        
        if tool_name in tool_map:
            tool_result = None
            tool_result_str = ""
            serializable_result = None

            try:
                tool_result = tool_map[tool_name](**arguments)
            except Exception as e:
                tool_result = f"Error executing tool '{tool_name}': {str(e)}"

            try:
                tool_result_str = json.dumps(tool_result, default=str)
                try:
                    serializable_result = json.loads(tool_result_str)
                except json.JSONDecodeError:
                    serializable_result = {"result": tool_result_str}
            except Exception as e_serialize:
                tool_result_str = f"Error serializing result for {tool_name}: {str(e_serialize)}"
                serializable_result = {"error": tool_result_str}

            result["tool_results"].append({
                "tool_call_id": tool_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "result": serializable_result
            })

            # Add tool result as a tool message (proper format for multi-turn)
            result["messages"].append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": tool_result_str
            })
    
    return result