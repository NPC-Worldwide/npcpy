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

def sanitize_messages(messages: list) -> list:
    if not messages:
        return messages

    valid_tool_call_ids = set()
    for msg in messages:
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            for tc in msg['tool_calls']:
                if isinstance(tc, dict):
                    tc_id = tc.get('id')
                else:
                    tc_id = getattr(tc, 'id', None)
                if tc_id:
                    valid_tool_call_ids.add(tc_id)

    cleaned = []
    for msg in messages:
        if msg.get('role') == 'tool':
            tc_id = msg.get('tool_call_id')
            if tc_id and tc_id not in valid_tool_call_ids:
                content = msg.get('content', '')
                name = msg.get('name', 'tool')
                cleaned.append({
                    'role': 'assistant',
                    'content': f"[{name} result]: {content}" if name != 'tool' else content
                })
                continue
        cleaned.append(msg)

    return cleaned


TOKEN_COSTS = {
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
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-3-pro": (2.00, 12.00),
    "llama-3": (0.05, 0.08),
    "llama-3.1": (0.05, 0.08),
    "llama-3.2": (0.05, 0.08),
    "llama-4": (0.05, 0.10),
    "mixtral": (0.24, 0.24),
    "deepseek-v3": (0.27, 1.10),
    "deepseek-r1": (0.55, 2.19),
    "mistral-large": (2.00, 6.00),
    "mistral-small": (0.20, 0.60),
    "grok-2": (2.00, 10.00),
    "grok-3": (3.00, 15.00),
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a response."""
    if not model:
        return 0.0

    model_key = model.split("/")[-1].lower()

    costs = None
    for key, cost in TOKEN_COSTS.items():
        if key in model_key or model_key in key:
            costs = cost
            break

    if not costs:
        return 0.0

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
                else:
                    text_extensions = {'.txt', '.text', '.log', '.md', '.markdown', '.rst', '.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.cfg', '.xml', '.html', '.htm', '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.h', '.cpp', '.hpp', '.go', '.rs', '.rb', '.php', '.sh', '.bash', '.sql', '.css', '.scss'}
                    filename = os.path.basename(attachment)
                    if ext in text_extensions or ext == '':
                        try:
                            with open(attachment, 'r', encoding='utf-8', errors='replace') as f:
                                text_content = f.read()
                            max_chars = 50000
                            if len(text_content) > max_chars:
                                text_content = text_content[:max_chars] + f"\n\n... [truncated]"
                            if text_content.strip():
                                if prompt:
                                    prompt += f"\n\nContent from {filename}:\n```\n{text_content}\n```"
                                else:
                                    prompt = f"Content from {filename}:\n```\n{text_content}\n```"
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
            result["response"] = res
            return result

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
        
        
        clean_messages = []
        tool_results_summary = []

        for msg in processed_result["messages"]:
            role = msg.get('role', '')
            if role == 'assistant' and 'tool_calls' in msg:
                continue
            elif role == 'tool':
                content = msg.get('content', '')
                if len(content) > 2000:
                    content = content[:2000] + "... (truncated)"
                tool_results_summary.append(content)
            else:
                clean_messages.append(msg)

        if tool_results_summary:
            clean_messages.append({
                "role": "assistant",
                "content": "I executed the requested tools. Here are the results:\n\n" + "\n\n".join(tool_results_summary)
            })

        clean_messages.append({
            "role": "user",
            "content": "Based on the tool results above, provide a brief summary of what happened. Do NOT output any code - the tool has already executed. Just describe the results concisely."
        })

        final_api_params = {
            "model": model,
            "messages": clean_messages,
            "stream": stream,
        }

        if stream:
            final_stream = ollama.chat(**final_api_params, options=options)
            processed_result["response"] = final_stream
        else:
            final_resp = ollama.chat(**final_api_params, options=options)
            final_message = final_resp.get("message", {})
            final_content = final_message.get("content", "")
            if final_content:
                processed_result["response"] = final_content
                processed_result["messages"].append({"role": "assistant", "content": final_content})
            elif tool_results_summary:
                processed_result["response"] = "\n\n".join(tool_results_summary)
            else:
                processed_result["response"] = "Tool executed successfully."

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

def get_lora_response(
    prompt: str = None,
    model: str = None,
    tools: list = None,
    tool_map: Dict = None,
    format: str = None,
    messages: List[Dict[str, str]] = None,
    stream: bool = False,
    auto_process_tool_calls: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate response using a LoRA adapter on top of a base model.
    The adapter path should contain adapter_config.json with base_model_name_or_path.
    """
    print(f"🎯 get_lora_response called with model={model}, prompt={prompt[:50] if prompt else 'None'}...")

    result = {
        "response": None,
        "messages": messages.copy() if messages else [],
        "raw_response": None,
        "tool_calls": [],
        "tool_results": []
    }

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        print("🎯 Successfully imported torch, transformers, peft")
    except ImportError as e:
        print(f"🎯 Import error: {e}")
        return {
            "response": "",
            "messages": messages or [],
            "error": f"Missing dependencies for LoRA. Install with: pip install transformers peft torch. Error: {e}"
        }

    adapter_path = os.path.expanduser(model)
    adapter_config_path = os.path.join(adapter_path, 'adapter_config.json')

    if not os.path.exists(adapter_config_path):
        return {
            "response": "",
            "messages": messages or [],
            "error": f"No adapter_config.json found at {adapter_path}"
        }

    try:
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_id = adapter_config.get('base_model_name_or_path')
        if not base_model_id:
            return {
                "response": "",
                "messages": messages or [],
                "error": "adapter_config.json missing base_model_name_or_path"
            }
    except Exception as e:
        return {
            "response": "",
            "messages": messages or [],
            "error": f"Failed to read adapter config: {e}"
        }

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

    try:
        logger.info(f"Loading base model: {base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading LoRA adapter: {adapter_path}")
        model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)

        chat_text = tokenizer.apply_chat_template(
            result["messages"],
            tokenize=False,
            add_generation_prompt=True
        )
        device = next(model_with_adapter.parameters()).device
        inputs = tokenizer(chat_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        max_new_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)

        with torch.no_grad():
            outputs = model_with_adapter.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_content = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        result["response"] = response_content
        result["raw_response"] = response_content
        result["messages"].append({"role": "assistant", "content": response_content})

        if format == "json":
            try:
                if response_content.startswith("```json"):
                    response_content = response_content.replace("```json", "").replace("```", "").strip()
                parsed_response = json.loads(response_content)
                result["response"] = parsed_response
            except json.JSONDecodeError:
                result["error"] = f"Invalid JSON response: {response_content}"

    except Exception as e:
        logger.error(f"LoRA inference error: {e}")
        result["error"] = f"LoRA inference error: {str(e)}"
        result["response"] = ""

    return result

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
        n_ctx = kwargs.get("n_ctx", 4096)
        n_gpu_layers = kwargs.get("n_gpu_layers", -1)

        llm = Llama(
            model_path=model,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )

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

_AIRLLM_MODEL_CACHE = {}
_AIRLLM_MLX_PATCHED = False

def _patch_airllm_mlx_bias():
    """
    Monkey-patch airllm's MLX Attention/FeedForward to use bias=True.
    AirLLM hardcodes bias=False which fails for non-Llama architectures (e.g. Qwen2).
    Using bias=True is safe: MLX nn.Linear(bias=True) accepts weight-only updates,
    so Llama models (no bias in weights) still work correctly.
    """
    global _AIRLLM_MLX_PATCHED
    if _AIRLLM_MLX_PATCHED:
        return
    try:
        import airllm.airllm_llama_mlx as mlx_mod
        import mlx.core as mx
        from mlx import nn

        class PatchedAttention(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.args = args
                self.n_heads = args.n_heads
                self.n_kv_heads = args.n_kv_heads
                self.repeats = self.n_heads // self.n_kv_heads
                self.scale = args.head_dim ** -0.5
                self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=True)
                self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
                self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
                self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=True)
                self.rope = nn.RoPE(
                    args.head_dim, traditional=args.rope_traditional, base=args.rope_theta
                )

            def __call__(self, x, mask=None, cache=None):
                B, L, D = x.shape
                queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
                queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
                keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
                values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

                def repeat(a):
                    a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
                    return a.reshape([B, self.n_heads, L, -1])
                keys, values = map(repeat, (keys, values))

                if cache is not None:
                    key_cache, value_cache = cache
                    queries = self.rope(queries, offset=key_cache.shape[2])
                    keys = self.rope(keys, offset=key_cache.shape[2])
                    keys = mx.concatenate([key_cache, keys], axis=2)
                    values = mx.concatenate([value_cache, values], axis=2)
                else:
                    queries = self.rope(queries)
                    keys = self.rope(keys)

                scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
                if mask is not None:
                    scores += mask
                weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
                output = (weights @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.wo(output), (keys, values)

        class PatchedFeedForward(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=True)
                self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=True)
                self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=True)

            def __call__(self, x):
                return self.w2(nn.silu(self.w1(x)) * self.w3(x))

        mlx_mod.Attention = PatchedAttention
        mlx_mod.FeedForward = PatchedFeedForward
        _AIRLLM_MLX_PATCHED = True
        logger.debug("Patched airllm MLX classes for bias support")
    except Exception as e:
        logger.warning(f"Failed to patch airllm MLX bias support: {e}")

def get_airllm_response(
    prompt: str = None,
    model: str = None,
    tools: list = None,
    tool_map: Dict = None,
    format: str = None,
    messages: List[Dict[str, str]] = None,
    auto_process_tool_calls: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate response using AirLLM for 70B+ model inference.
    Supports macOS (MLX backend) and Linux (CUDA backend with 4-bit compression).
    """
    import platform
    is_macos = platform.system() == "Darwin"

    result = {
        "response": None,
        "messages": messages.copy() if messages else [],
        "raw_response": None,
        "tool_calls": [],
        "tool_results": []
    }

    try:
        from airllm import AutoModel
    except ImportError:
        result["response"] = ""
        result["error"] = "airllm not installed. Install with: pip install airllm"
        return result

    if is_macos:
        _patch_airllm_mlx_bias()

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

    model_name = model or "meta-llama/Meta-Llama-3.1-70B-Instruct"
    default_compression = None if is_macos else "4bit"
    compression = kwargs.get("compression", default_compression)
    max_tokens = kwargs.get("max_tokens", 256)
    temperature = kwargs.get("temperature", 0.7)

    hf_token = kwargs.get("hf_token")
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        try:
            from huggingface_hub import HfFolder
            hf_token = HfFolder.get_token()
        except Exception:
            pass

    cache_key = f"{model_name}:{compression}"
    if cache_key not in _AIRLLM_MODEL_CACHE:
        load_kwargs = {"pretrained_model_name_or_path": model_name}
        if compression:
            load_kwargs["compression"] = compression
        if hf_token:
            load_kwargs["hf_token"] = hf_token
        for k in ["delete_original", "max_seq_len", "prefetching"]:
            if k in kwargs:
                load_kwargs[k] = kwargs[k]
        _AIRLLM_MODEL_CACHE[cache_key] = AutoModel.from_pretrained(**load_kwargs)

    air_model = _AIRLLM_MODEL_CACHE[cache_key]

    try:
        chat_text = air_model.tokenizer.apply_chat_template(
            result["messages"], tokenize=False, add_generation_prompt=True
        )
    except Exception:
        chat_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in result["messages"]
        )
        chat_text += "\nassistant:"

    try:
        if is_macos:
            import mlx.core as mx
            tokens = air_model.tokenizer(
                chat_text, return_tensors="np", truncation=True, max_length=2048
            )
            output = air_model.generate(
                mx.array(tokens['input_ids']),
                max_new_tokens=max_tokens,
            )
            response_content = output if isinstance(output, str) else str(output)
        else:
            tokens = air_model.tokenizer(
                chat_text, return_tensors="pt", truncation=True, max_length=2048
            )
            gen_out = air_model.generate(
                tokens['input_ids'].cuda(),
                max_new_tokens=max_tokens,
            )
            output_ids = gen_out.sequences[0] if hasattr(gen_out, 'sequences') else gen_out[0]
            response_content = air_model.tokenizer.decode(output_ids, skip_special_tokens=True)
            input_text = air_model.tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
            if response_content.startswith(input_text):
                response_content = response_content[len(input_text):]

        response_content = response_content.strip()
        for stop_tok in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"]:
            if stop_tok in response_content:
                response_content = response_content[:response_content.index(stop_tok)].strip()
    except Exception as e:
        logger.error(f"AirLLM inference error: {e}")
        result["error"] = f"AirLLM inference error: {str(e)}"
        result["response"] = ""
        return result

    result["response"] = response_content
    result["raw_response"] = response_content
    result["messages"].append({"role": "assistant", "content": response_content})

    if format == "json":
        try:
            if response_content.startswith("```json"):
                response_content = response_content.replace("```json", "").replace("```", "").strip()
            parsed_response = json.loads(response_content)
            result["response"] = parsed_response
        except json.JSONDecodeError:
            result["error"] = f"Invalid JSON response: {response_content}"

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
    elif provider == 'transformers':
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
    elif provider == 'lora':
        print(f"🔧 LoRA provider detected, calling get_lora_response with model: {model}")
        result = get_lora_response(
            prompt=prompt,
            model=model,
            tools=tools,
            tool_map=tool_map,
            format=format,
            messages=messages,
            stream=stream,
            auto_process_tool_calls=auto_process_tool_calls,
            **kwargs
        )
        print(f"🔧 LoRA response: {result.get('response', 'NO RESPONSE')[:200] if result.get('response') else 'EMPTY'}")
        if result.get('error'):
            print(f"🔧 LoRA error: {result.get('error')}")
        return result
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
    elif provider == 'airllm':
        return get_airllm_response(
            prompt=prompt,
            model=model,
            tools=tools,
            tool_map=tool_map,
            format=format,
            messages=messages,
            auto_process_tool_calls=auto_process_tool_calls,
            **kwargs
        )
    elif provider == 'lmstudio' or (model and '.lmstudio' in str(model)):
        api_url = api_url or "http://127.0.0.1:1234/v1"
        provider = "openai"
        api_key = api_key or "lm-studio"
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 300
    elif provider == 'llamacpp-server':
        api_url = api_url or "http://127.0.0.1:8080/v1"
        provider = "openai"
        api_key = api_key or "llamacpp"
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 300

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
                else:
                    text_extensions = {'.txt', '.text', '.log', '.md', '.markdown', '.rst', '.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.cfg', '.xml', '.html', '.htm', '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.h', '.cpp', '.hpp', '.go', '.rs', '.rb', '.php', '.sh', '.bash', '.sql', '.css', '.scss'}
                    filename = os.path.basename(attachment)
                    if ext in text_extensions or ext == '':
                        try:
                            with open(attachment, 'r', encoding='utf-8', errors='replace') as f:
                                text_content = f.read()
                            max_chars = 50000
                            if len(text_content) > max_chars:
                                text_content = text_content[:max_chars] + f"\n\n... [truncated]"
                            if text_content.strip():
                                if prompt:
                                    prompt += f"\n\nContent from {filename}:\n```\n{text_content}\n```"
                                else:
                                    prompt = f"Content from {filename}:\n```\n{text_content}\n```"
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

    

    result["messages"] = sanitize_messages(result["messages"])
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

    if "api_base" in api_params and provider == "openai":
        api_params["model"] = f"openai/{model}"
    elif "/" not in model or model.startswith("/"):
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

        if hasattr(resp, 'usage') and resp.usage:
            result["usage"] = {
                "input_tokens": getattr(resp.usage, 'prompt_tokens', 0) or 0,
                "output_tokens": getattr(resp.usage, 'completion_tokens', 0) or 0,
            }
        elif hasattr(resp, 'prompt_eval_count'):
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

                        if '```yaml' in llm_response:
                            start = llm_response.find('```yaml') + 7
                            end = llm_response.rfind('```')
                            if end > start:
                                llm_response = llm_response[start:end].strip()
                        elif '```' in llm_response:
                            start = llm_response.find('```') + 3
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

        clean_messages = []
        tool_results_summary = []

        for msg in processed_result["messages"]:
            role = msg.get('role', '')
            if role == 'assistant' and 'tool_calls' in msg:
                continue
            elif role == 'tool':
                content = msg.get('content', '')
                if len(content) > 2000:
                    content = content[:2000] + "... (truncated)"
                tool_results_summary.append(content)
            else:
                clean_messages.append(msg)

        if tool_results_summary:
            clean_messages.append({
                "role": "assistant",
                "content": "I executed the requested tools. Here are the results:\n\n" + "\n\n".join(tool_results_summary)
            })

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
                if tool_results_summary:
                    fallback_content = "\n\n".join(tool_results_summary)
                else:
                    fallback_content = "Tool executed successfully."
                processed_result["response"] = fallback_content
                processed_result["messages"].append({"role": "assistant", "content": fallback_content})

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

    tool_calls_for_message = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            tool_calls_for_message.append(tc)
        else:
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

            result["messages"].append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": tool_result_str
            })
    
    return result