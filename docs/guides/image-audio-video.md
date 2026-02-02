# Image, Audio & Video Generation

npcpy provides unified functions for generating images, audio, and video across multiple providers. All generation functions live under `npcpy.llm_funcs` (high-level wrappers) and `npcpy.gen.*` (provider-specific implementations).

## Image Generation

### gen_image

The top-level function for image generation and editing.

```python
from npcpy.llm_funcs import gen_image
```

**Signature:**

```python
def gen_image(
    prompt: str,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    height: int = 1024,
    width: int = 1024,
    n_images: int = 1,
    input_images: List[Union[str, bytes, PIL.Image.Image]] = None,
    save: bool = False,
    filename: str = '',
    api_key: str = None,
) -> List[PIL.Image.Image]
```

**Returns:** A list of PIL Image objects.

### Ollama

Generate images locally with Ollama image generation models.

```python
from npcpy.llm_funcs import gen_image

images = gen_image(
    "A serene mountain lake at sunset",
    model="x/z-image-turbo",
    provider="ollama",
    height=512,
    width=512,
)
images[0].show()
```

Supported Ollama image models:
- `x/z-image-turbo` (default for Ollama)
- `x/flux2-klein`

Make sure the model is pulled first: `ollama pull x/z-image-turbo`

### Diffusers

Run Stable Diffusion and other HuggingFace diffusion models locally.

```python
images = gen_image(
    "A cyberpunk cityscape at night",
    model="runwayml/stable-diffusion-v1-5",
    provider="diffusers",
    height=512,
    width=512,
)
images[0].save("cityscape.png")
```

The diffusers backend enables memory optimizations automatically (attention slicing, CPU offload, xformers when available). For fine-tuned models, pass the local directory path as the model name:

```python
images = gen_image(
    "My custom style",
    model="/path/to/my-finetuned-model",
    provider="diffusers",
)
```

### OpenAI

Generate images with DALL-E 2, DALL-E 3, or GPT Image 1.

```python
images = gen_image(
    "An oil painting of a cat wearing a top hat",
    model="dall-e-2",
    provider="openai",
    height=1024,
    width=1024,
    n_images=2,
)
```

### Gemini

Generate images using Google's Gemini models.

```python
images = gen_image(
    "A watercolor painting of cherry blossoms",
    model="gemini-2.5-flash-image",
    provider="gemini",
)
```

For Imagen models:

```python
images = gen_image(
    "A photorealistic landscape",
    model="imagen-3.0-generate-002",
    provider="gemini",
    n_images=2,
)
```

### Image Editing

Edit existing images by passing attachments. The attachments parameter accepts file paths, raw bytes, or PIL Image objects.

```python
# OpenAI image editing with gpt-image-1
images = gen_image(
    "Add a rainbow in the sky",
    model="gpt-image-1",
    provider="openai",
    input_images=["photo.png"],
)

# Gemini image editing
images = gen_image(
    "Make the sky more dramatic",
    model="gemini-2.5-flash-image",
    provider="gemini",
    input_images=["landscape.jpg"],
)
```

There is also a convenience function for editing:

```python
from npcpy.gen.image_gen import edit_image

edited = edit_image(
    prompt="Add a rainbow in the sky",
    image_path="photo.png",
    provider="openai",
    model="gpt-image-1",
    save_path="edited.png",
)
```

### Saving Images

Use the `save` and `filename` parameters:

```python
images = gen_image(
    "A fractal pattern",
    model="x/z-image-turbo",
    provider="ollama",
    save=True,
    filename="fractal",
)
# Saves fractal_0.png, fractal_1.png, etc.
```

Or save manually:

```python
images = gen_image("A sunset", model="dall-e-2", provider="openai")
images[0].save("sunset.png")
```

## Video Generation

### gen_video

Generate video from text prompts.

```python
from npcpy.llm_funcs import gen_video
```

**Signature:**

```python
def gen_video(
    prompt: str,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    device: str = "cpu",
    output_path: str = "",
    num_inference_steps: int = 10,
    num_frames: int = 25,
    height: int = 256,
    width: int = 256,
    negative_prompt: str = "",
    messages: list = None,
) -> dict
```

### Diffusers Backend

Uses the `damo-vilab/text-to-video-ms-1.7b` model by default.

```python
result = gen_video(
    "A timelapse of flowers blooming",
    provider="diffusers",
    device="cpu",
    num_frames=25,
    height=256,
    width=256,
)
print(result["output"])  # path to the generated .mp4
```

### Gemini / Veo 3

Generate high-fidelity video with synchronized audio using Google's Veo 3 API.

```python
result = gen_video(
    "A drone shot over a tropical island with waves crashing",
    model="veo-3.0-generate-preview",
    provider="gemini",
    negative_prompt="blurry, low quality",
)
print(result["output"])  # path to the generated .mp4
```

Videos are saved to `~/.npcsh/videos/` by default.

## Audio Generation (Text-to-Speech)

npcpy provides a unified TTS interface supporting multiple engines.

```python
from npcpy.gen.audio_gen import text_to_speech
```

### Unified Interface

```python
def text_to_speech(
    text: str,
    engine: str = "kokoro",
    voice: str = None,
    **kwargs,
) -> bytes
```

Returns raw audio bytes. The format depends on the engine (WAV for Kokoro/Qwen3, MP3 for ElevenLabs/gTTS, PCM16 for OpenAI/Gemini).

### Kokoro (Local Neural TTS)

Default engine. Runs locally with no API key needed.

```python
from npcpy.gen.audio_gen import text_to_speech

audio_bytes = text_to_speech(
    "Hello, welcome to npcpy!",
    engine="kokoro",
    voice="af_heart",   # female American voice
    speed=1.0,
)

# Save to file
with open("greeting.wav", "wb") as f:
    f.write(audio_bytes)
```

Available Kokoro voices: `af_heart`, `af_bella`, `af_sarah`, `af_nicole`, `af_sky` (female American), `am_adam`, `am_michael` (male American), `bf_emma`, `bf_isabella` (female British), `bm_george`, `bm_lewis` (male British).

Install: `pip install kokoro soundfile`

### Qwen3-TTS (Local High-Quality Multilingual)

High-quality local TTS with voice cloning and voice design capabilities.

```python
audio_bytes = text_to_speech(
    "Welcome to the future of AI.",
    engine="qwen3",
    voice="ryan",
    language="english",
    model_size="1.7B",
)
```

Available voices: `aiden`, `dylan`, `eric`, `ryan` (male), `serena`, `vivian`, `sohee`, `ono_anna` (female), `uncle_fu` (male).

Voice cloning from a reference audio file:

```python
from npcpy.gen.audio_gen import tts_qwen3

audio_bytes = tts_qwen3(
    "Clone this voice.",
    ref_audio="reference.wav",
    ref_text="The transcript of the reference audio.",
)
```

### ElevenLabs (Cloud)

High-quality cloud TTS. Requires `ELEVENLABS_API_KEY` environment variable.

```python
audio_bytes = text_to_speech(
    "This is ElevenLabs quality speech.",
    engine="elevenlabs",
)

# Or use the direct function
from npcpy.gen.audio_gen import tts_elevenlabs

audio_bytes = tts_elevenlabs(
    "Hello world!",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2",
)
```

For lowest-latency streaming via WebSocket:

```python
import asyncio
from npcpy.gen.audio_gen import tts_elevenlabs_stream

audio_bytes = asyncio.run(tts_elevenlabs_stream(
    "Stream this text.",
    model_id="eleven_turbo_v2_5",
))
```

### OpenAI Realtime Voice

Uses the OpenAI Realtime API. Requires `OPENAI_API_KEY`.

```python
audio_bytes = text_to_speech(
    "Hello from OpenAI!",
    engine="openai",
    voice="alloy",   # alloy, echo, shimmer, ash, ballad, coral, sage, verse
)
```

### Gemini Live

Uses the Gemini Live API. Requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

```python
audio_bytes = text_to_speech(
    "Hello from Gemini!",
    engine="gemini",
    voice="Puck",   # Puck, Charon, Kore, Fenrir, Aoede
)
```

### gTTS (Free Fallback)

Google Text-to-Speech. No API key needed, but lower quality.

```python
audio_bytes = text_to_speech(
    "Simple fallback TTS.",
    engine="gtts",
    voice="en",   # language code
)
```

### Discovering Available Engines and Voices

```python
from npcpy.gen.audio_gen import get_available_engines, get_available_voices

# Check which engines are installed and configured
engines = get_available_engines()
for name, info in engines.items():
    status = "available" if info["available"] else "not available"
    print(f"{info['name']}: {status} ({info['description']})")

# List voices for a specific engine
voices = get_available_voices("kokoro")
for v in voices:
    print(f"  {v['id']}: {v['name']} ({v['gender']})")
```

### Audio Utilities

Convert between audio formats:

```python
from npcpy.gen.audio_gen import pcm16_to_wav, wav_to_pcm16, audio_to_base64

# PCM16 (from OpenAI/Gemini) to WAV
wav_bytes = pcm16_to_wav(pcm_data, sample_rate=24000)

# WAV to PCM16
pcm_data, sample_rate = wav_to_pcm16(wav_bytes)

# Encode for transmission
b64_string = audio_to_base64(audio_bytes)
```
