import os
import base64
import io
from typing import Union, List, Optional

import PIL
from PIL import Image

from litellm import image_generation
import requests
from urllib.request import urlopen


def generate_image_diffusers(
    prompt: str,
    model: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cpu",
    height: int = 512,
    width: int = 512,
):
    """Generate an image using the Stable Diffusion API with memory optimization."""
    import torch
    import gc
    import os
    from diffusers import DiffusionPipeline, StableDiffusionPipeline

    try:
        torch_dtype = torch.float16 if device != "cpu" and torch.cuda.is_available() else torch.float32
        
        if os.path.isdir(model):
            print(f"🌋 Loading fine-tuned Diffusers model from local path: {model}")
            
            checkpoint_path = os.path.join(model, 'model_final.pt')
            if os.path.exists(checkpoint_path):
                print(f"🌋 Found model_final.pt at {checkpoint_path}.")
                
                # Load checkpoint to inspect it
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                
                # Check if this is a custom SimpleUNet model (from your training code)
                # vs a Stable Diffusion UNet2DConditionModel
                if 'config' in checkpoint and hasattr(checkpoint['config'], 'image_size'):
                    print(f"🌋 Detected custom SimpleUNet model, using custom generation")
                    # Use your custom generate_image function from npcpy.ft.diff
                    from npcpy.ft.diff import generate_image as custom_generate_image
                    
                    # Your custom model ignores prompts and generates based on training data
                    image = custom_generate_image(
                        model_path=checkpoint_path,
                        prompt=prompt,
                        num_samples=1,
                        image_size=height  # Use the requested height
                    )
                    return image
                
                else:
                    # This is a Stable Diffusion checkpoint
                    print(f"🌋 Detected Stable Diffusion UNet checkpoint")
                    base_model_id = "runwayml/stable-diffusion-v1-5"
                    print(f"🌋 Loading base pipeline: {base_model_id}")
                    pipe = StableDiffusionPipeline.from_pretrained(
                        base_model_id,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        variant="fp16" if torch_dtype == torch.float16 else None,
                    )
                    
                    print(f"🌋 Loading custom UNet weights from {checkpoint_path}")
                    
                    # Extract the actual model state dict
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        unet_state_dict = checkpoint['model_state_dict']
                        print(f"🌋 Extracted model_state_dict from checkpoint")
                    else:
                        unet_state_dict = checkpoint
                        print(f"🌋 Using checkpoint directly as state_dict")
                    
                    # Load the state dict into the UNet
                    pipe.unet.load_state_dict(unet_state_dict)
                    pipe = pipe.to(device)
                    print(f"🌋 Successfully loaded fine-tuned UNet weights")
                    
            else:
                raise OSError(f"Error: Fine-tuned model directory {model} does not contain 'model_final.pt'")

        else:
            print(f"🌋 Loading standard Diffusers model: {model}")
            if 'Qwen' in model:
                pipe = DiffusionPipeline.from_pretrained(
                    model,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    variant="fp16" if torch_dtype == torch.float16 else None,
                )
            else:        
                pipe = StableDiffusionPipeline.from_pretrained(
                    model,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    variant="fp16" if torch_dtype == torch.float16 else None,
                )
        
        # Common pipeline setup for Stable Diffusion models
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
        
        if hasattr(pipe, 'enable_model_cpu_offload') and device != "cpu":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
        
        if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        
        with torch.inference_mode():
            result = pipe(prompt, height=height, width=width, num_inference_steps=20)
            image = result.images[0]
        
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return image
        
    except Exception as e:
        if 'pipe' in locals():
            del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        if "out of memory" in str(e).lower() or "killed" in str(e).lower():
            print(f"Memory error during image generation: {e}")
            print("Suggestions:")
            print("1. Try using a smaller model like 'CompVis/stable-diffusion-v1-4'")
            print("2. Reduce image dimensions (e.g., height=256, width=256)")
            print("3. Use a cloud service for image generation")
            raise MemoryError(f"Insufficient memory for image generation with model {model}. Try a smaller model or reduce image size.")
        else:
            raise e
import os
import base64
import io
from typing import Union, List, Optional

import PIL
from PIL import Image

import requests
from urllib.request import urlopen

def openai_image_gen(
    prompt: str,
    model: str = "dall-e-2",
    attachments: Union[List[Union[str, bytes, Image.Image]], None] = None,
    height: int = 1024,
    width: int = 1024,
    n_images: int = 1,
    api_key: Optional[str] = None,
):
    """Generate or edit an image using the OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    if height is None:
        height = 1024
    if width is None:
        width = 1024

    size_str = f"{width}x{height}"

    if attachments is not None:
        processed_images = []
        files_to_close = []
        for attachment in attachments:
            if isinstance(attachment, str):
                file_handle = open(attachment, "rb")
                processed_images.append(file_handle)
                files_to_close.append(file_handle)
            elif isinstance(attachment, bytes):
                img_byte_arr = io.BytesIO(attachment)
                img_byte_arr.name = 'image.png'  # FIX: Add filename hint
                processed_images.append(img_byte_arr)
            elif isinstance(attachment, Image.Image):
                img_byte_arr = io.BytesIO()
                attachment.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                img_byte_arr.name = 'image.png'  # FIX: Add filename hint
                processed_images.append(img_byte_arr)
        
        try:
            result = client.images.edit(
                model=model,
                image=processed_images[0],
                prompt=prompt,
                n=n_images,
                size=size_str,
            )
        finally:
            # This ensures any files we opened are properly closed
            for f in files_to_close:
                f.close()
    else:
        result = client.images.generate(
            model=model,
            prompt=prompt,
            n=n_images,
            size=size_str,
        )

    collected_images = []
    for item_data in result.data:
        if model == 'gpt-image-1':
            image_base64 = item_data.b64_json
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
        elif model == 'dall-e-2' or model == 'dall-e-3':
            image_url = item_data.url
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
        else:
            image = item_data
        collected_images.append(image)
        
    return collected_images



def gemini_image_gen(
    prompt: str,
    model: str = "gemini-2.5-flash",
    attachments: Union[List[Union[str, bytes, Image.Image]], None] = None,
    n_images: int = 1,
    api_key: Optional[str] = None,
):
    """Generate or edit images using Google's Gemini API."""
    from google import genai
    from google.genai import types
    from io import BytesIO
    import requests

    if api_key is None:
        api_key = os.environ.get('GEMINI_API_KEY')
    
    client = genai.Client(api_key=api_key)
    collected_images = []

    if attachments is not None:
        processed_contents = [prompt]
        
        for attachment in attachments:
            if isinstance(attachment, str):
                if attachment.startswith(('http://', 'https://')):
                    image_bytes = requests.get(attachment).content
                    mime_type = 'image/jpeg'
                    processed_contents.append(
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type
                        )
                    )
                else:
                    with open(attachment, 'rb') as f:
                        image_bytes = f.read()
                    
                    if attachment.lower().endswith('.png'):
                        mime_type = 'image/png'
                    elif attachment.lower().endswith('.jpg') or attachment.lower().endswith('.jpeg'):
                        mime_type = 'image/jpeg'
                    elif attachment.lower().endswith('.webp'):
                        mime_type = 'image/webp'
                    else:
                        mime_type = 'image/jpeg'
                    
                    processed_contents.append(
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type
                        )
                    )
            elif isinstance(attachment, bytes):
                processed_contents.append(
                    types.Part.from_bytes(
                        data=attachment,
                        mime_type='image/jpeg'
                    )
                )
            elif isinstance(attachment, Image.Image):
                img_byte_arr = BytesIO()
                attachment.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                processed_contents.append(
                    types.Part.from_bytes(
                        data=img_byte_arr.getvalue(),
                        mime_type='image/png'
                    )
                )
        
        response = client.models.generate_content(
            model=model,
            contents=processed_contents,
        )
        
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        image_data = part.inline_data.data
                        collected_images.append(Image.open(BytesIO(image_data)))
        
        if not collected_images and hasattr(response, 'text'):
            print(f"Gemini response text: {response.text}")
        
        return collected_images
    else:
        if 'imagen' in model:
            response = client.models.generate_images(
                model=model,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=n_images,
                )
            )
            for generated_image in response.generated_images:
                collected_images.append(Image.open(BytesIO(generated_image.image.image_bytes)))
            return collected_images
            
        elif 'flash-image' in model or 'image-preview' in model or '2.5-flash' in model:
            response = client.models.generate_content(
                model=model,
                contents=[prompt],
            )
            
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            image_data = part.inline_data.data
                            collected_images.append(Image.open(BytesIO(image_data)))
            
            if not collected_images and hasattr(response, 'text'):
                print(f"Gemini response text: {response.text}")
            
            return collected_images
        
        else:
            raise ValueError(f"Unsupported Gemini image model or API usage for new generation: '{model}'")
# In npcpy/gen/image_gen.py, find the generate_image function and replace it with this:

def generate_image(
    prompt: str,
    model: str ,
    provider: str ,
    height: int = 1024,
    width: int = 1024,
    n_images: int = 1,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    attachments: Union[List[Union[str, bytes, Image.Image]], None] = None,
    save_path: Optional[str] = None,
    custom_model_path: Optional[str] = None, # <--- NEW: Accept custom_model_path,
    
):
    """
    Unified function to generate or edit images using various providers.
    
    Args:
        prompt (str): The prompt for generating/editing the image.
        model (str): The model to use.
        provider (str): The provider to use ('openai', 'diffusers', 'gemini').
        height (int): The height of the output image.
        width (int): The width of the output image.
        n_images (int): Number of images to generate.
        api_key (str): API key for the provider.
        api_url (str): API URL for the provider.
        attachments (list): List of images for editing. Can be file paths, bytes, or PIL Images.
        save_path (str): Path to save the generated image.
        custom_model_path (str): Path to a locally fine-tuned Diffusers model. <--- NEW
        
    Returns:
        List[PIL.Image.Image]: A list of generated PIL Image objects.
    """
    from urllib.request import urlopen
    import os # Ensure os is imported for path checks

    if model is None and custom_model_path is None: # Only set default if no model or custom path is provided
        if provider == "openai":
            model = "dall-e-2"
        elif provider == "diffusers":
            model = "runwayml/stable-diffusion-v1-5"
        elif provider == "gemini":
            model = "gemini-2.5-flash-image-preview"
    
    all_generated_pil_images = []

    # <--- CRITICAL FIX: Handle custom_model_path for Diffusers here
    if provider == "diffusers":
        # If a custom_model_path is provided and exists, use it instead of a generic model name
        if custom_model_path and os.path.isdir(custom_model_path):
            print(f"🌋 Using custom Diffusers model from path: {custom_model_path}")
            model_to_use = custom_model_path
        else:
            # Otherwise, use the standard model name (e.g., "runwayml/stable-diffusion-v1-5")
            model_to_use = model
            print(f"🌋 Using standard Diffusers model: {model_to_use}")

        for _ in range(n_images):
            try:
                image = generate_image_diffusers(
                    prompt=prompt, 
                    model=model_to_use, # <--- Pass the resolved model_to_use
                    height=height, 
                    width=width
                )
                all_generated_pil_images.append(image)
            except MemoryError as e:
                raise e
            except Exception as e:
                raise e

    elif provider == "openai":
        images = openai_image_gen(
            prompt=prompt,
            model=model,
            attachments=attachments,
            height=height,
            width=width,
            n_images=n_images,
            api_key=api_key
            
        )
        all_generated_pil_images.extend(images)

    elif provider == "gemini":
        images = gemini_image_gen(
            prompt=prompt,
            model=model,
            attachments=attachments,
            n_images=n_images,
            api_key=api_key
        )
        all_generated_pil_images.extend(images)

    else:
        # This is the fallback for other providers or if provider is not explicitly handled
        valid_sizes = ["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"]
        size = f"{width}x{height}"
        
        if attachments is not None:
            raise ValueError("Image editing not supported with litellm provider")
        
        # The litellm.image_generation function expects the provider as part of the model string
        # e.g., "huggingface/starcoder" or "openai/dall-e-3"
        # Since we've already handled "diffusers", "openai", "gemini" above,
        # this 'else' block implies a generic litellm call.
        # We need to ensure the model string is correctly formatted for litellm.
        # However, the error message "LLM Provider NOT provided" suggests litellm
        # is not even getting the `provider` correctly.
        # The fix for this is ensuring the `provider` is explicitly passed to litellm.image_generation
        # which is already happening in `gen_image` in `llm_funcs.py`
        
        # If we reach here, it means the provider is not 'diffusers', 'openai', or 'gemini',
        # and litellm is the intended route. We need to pass the provider explicitly.
        # The original code here was trying to construct `model=f"{provider}/{model}"`
        # but the error indicates `provider` itself was missing.
        # The `image_generation` from litellm expects `model` to be `provider/model_name`.
        # Since the `provider` variable is available, we can construct this.
        
        # This block is for generic litellm providers (not diffusers, openai, gemini)
        # The error indicates `provider` itself was not making it to litellm.
        # This `generate_image` function already receives `provider`.
        # The issue is likely how `gen_image` in `llm_funcs.py` calls this `generate_image`.
        # However, if this `else` branch is hit, we ensure litellm gets the provider.
        
        # Construct the model string for litellm
        litellm_model_string = f"{provider}/{model}" if provider and model else model
        
        image_response = image_generation(
            prompt=prompt,
            model=litellm_model_string, # <--- Ensure model string includes provider for litellm
            n=n_images,
            size=size,
            api_key=api_key,
            api_base=api_url,
        )
        for item_data in image_response.data:
            if hasattr(item_data, 'url') and item_data.url:
                with urlopen(item_data.url) as response:
                    all_generated_pil_images.append(Image.open(response))
            elif hasattr(item_data, 'b64_json') and item_data.b64_json:
                image_bytes = base64.b64decode(item_data.b64_json)
                all_generated_pil_images.append(Image.open(io.BytesIO(image_bytes)))
            else:
                print(f"Warning: litellm ImageResponse item has no URL or b64_json: {item_data}")

    if save_path:
        for i, img_item in enumerate(all_generated_pil_images):
            temp_save_path = f"{os.path.splitext(save_path)[0]}_{i}{os.path.splitext(save_path)[1]}"
            if isinstance(img_item, Image.Image):
                img_item.save(temp_save_path)
            else:
                print(f"Warning: Attempting to save non-PIL image item: {type(img_item)}. Skipping save for this item.")

    return all_generated_pil_images

def edit_image(
    prompt: str,
    image_path: str,
    provider: str = "openai",
    model: str = None,
    height: int = 1024,
    width: int = 1024,
    save_path: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Convenience function to edit an existing image.
    
    Args:
        prompt (str): The editing instructions.
        image_path (str): Path to the image to edit.
        provider (str): The provider to use for editing.
        model (str): The model to use.
        height (int): The height of the output image.
        width (int): The width of the output image.
        save_path (str): Path to save the edited image.
        api_key (str): API key for the provider.
        
    Returns:
        PIL.Image: The edited image.
    """
    if model is None:
        if provider == "openai":
            model = "gpt-image-1"
        elif provider == "gemini":
            model = "gemini-2.5-flash-image-preview"
    
    image = generate_image(
        prompt=prompt,
        provider=provider,
        model=model,
        attachments=[image_path],
        height=height,
        width=width,
        save_path=save_path,
        api_key=api_key
    )
    
    return image