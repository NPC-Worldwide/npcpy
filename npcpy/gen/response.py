from typing import Any, Dict, List, Union
from pydantic import BaseModel
from npcpy.data.image import compress_image
from npcpy.npc_sysenv import get_system_message, lookup_provider, render_markdown
import base64
import json
import uuid
import os 
try: 
    import ollama
except ImportError:
    pass
except OSError:
    
    print("Ollama is not installed or not available. Please install it to use this feature.")
try:
    import litellm
    from litellm import completion
except ImportError:
    pass
except OSError:
    
    pass

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
           result = process_tool_calls(result, tool_map, "local", "transformers", result["messages"])
   
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
                        
    if image_paths:
        last_user_idx = -1
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                last_user_idx = i
        if last_user_idx == -1:
            messages.append({"role": "user", "content": ""})
            last_user_idx = len(messages) - 1
        messages[last_user_idx]["images"] = image_paths

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
            
            return result

    
    
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
                                              stream=False)
        
        
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
        
        return result
    
import time 


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

    if api_url is not None and (provider == "openai-like" or provider == "openai"):
        api_params["api_base"] = api_url
        provider = "openai"
    
    
    if provider =='enpisi' and api_url is None:
        api_params['api_base'] = 'https://api.enpisi.com'
        if api_key is None:
            api_key = os.environ.get('NPC_STUDIO_LICENSE')
            api_params['api_key'] = api_key
        if '-npc' in model: 
            model = model.split('-npc')[0]
        provider = "openai"

    if isinstance(format, BaseModel):
        api_params["response_format"] = format
    if model is None:
        model = os.environ.get("NPCSH_CHAT_MODEL", "llama3.2")
    if provider is None:
        provider = os.environ.get("NPCSH_CHAT_PROVIDER")

    api_params["model"] = f"{provider}/{model}" if "/" not in model else model
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
                "response_format", "user",
            ]:
                api_params[key] = value

    if not auto_process_tool_calls or not (tools and tool_map):
        api_params["stream"] = stream
        resp = completion(**api_params)
        result["raw_response"] = resp
        
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
                        if llm_response.startswith("```json"):
                            llm_response = llm_response.replace("```json", "").replace("```", "").strip()
                        parsed_json = json.loads(llm_response)
                        
                        if "json" in parsed_json:
                            result["response"] = parsed_json["json"]
                        else:
                            result["response"] = parsed_json
                        
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"JSON parsing error: {str(e)}")
                    print(f"Raw response: {llm_response}")
                    result["error"] = "Invalid JSON response"
            
            return result

    
    
    initial_api_params = api_params.copy()
    initial_api_params["stream"] = False
    
    
    resp = completion(**initial_api_params)
    result["raw_response"] = resp
    
    
    has_tool_calls = hasattr(resp.choices[0].message, 'tool_calls') and resp.choices[0].message.tool_calls
    
    if has_tool_calls:

        
        result["tool_calls"] = resp.choices[0].message.tool_calls
        
        
        processed_result = process_tool_calls(result, 
                                              tool_map, 
                                              model, 
                                              provider, 
                                              result["messages"], 
                                              stream=False)
        
        
        if stream:

            

            clean_messages = []
            for msg in processed_result["messages"]:
                if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                    continue  
                                
                else:
                    clean_messages.append(msg)
            
            final_api_params = api_params.copy()
            final_api_params["messages"] = clean_messages
            final_api_params["stream"] = True


            final_api_params = api_params.copy()
            final_api_params["messages"] = clean_messages
            final_api_params["stream"] = True
            if "tools" in final_api_params:
                del final_api_params["tools"]
            if "tool_choice" in final_api_params:
                del final_api_params["tool_choice"]

            final_stream = completion(**final_api_params)

            
            final_stream = completion(**final_api_params)
            processed_result["response"] = final_stream
            
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
def process_tool_calls(response_dict, tool_map, model, provider, messages, stream=False):
    result = response_dict.copy()
    result["tool_results"] = []
    
    if "messages" not in result:
        result["messages"] = messages if messages else []
    
    tool_calls = result.get("tool_calls", [])
    
    if not tool_calls:
        return result
    
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
                tool_result = f"Error executing tool '{tool_name}': {str(e)}. Tool map is : {tool_map}"

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
                "role": "assistant",
                "content": f'The results of the tool call for {tool_name} with {arguments} are as follows:' +tool_result_str
            })
    
    return result