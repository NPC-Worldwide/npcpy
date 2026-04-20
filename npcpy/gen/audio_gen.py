"""
Audio Generation Module for NPC
Supports multiple TTS engines including real-time voice APIs.

TTS Engines:
- Kokoro: Local neural TTS (default)
- Qwen3-TTS: Local high-quality multilingual TTS (0.6B/1.7B)
- ElevenLabs: Cloud TTS with streaming
- OpenAI: Realtime voice API
- Gemini: Live API for real-time voice
- gTTS: Google TTS fallback

Usage:
    from npcpy.gen.audio_gen import text_to_speech

    audio = text_to_speech("Hello world", engine="kokoro", voice="af_heart")
    audio = text_to_speech("Hello world", engine="qwen3", voice="ryan")

For STT, see npcpy.data.audio
"""

import os
import io
import base64
import json
import asyncio
import tempfile
from typing import Optional, Callable, Any

def tts_kokoro(
    text: str,
    voice: str = "af_heart",
    lang_code: str = "a",
    speed: float = 1.0
) -> bytes:
    """
    Generate speech using Kokoro local neural TTS.

    Args:
        text: Text to synthesize
        voice: Voice ID (af_heart, am_adam, bf_emma, etc.)
        lang_code: 'a' for American, 'b' for British
        speed: Speech speed multiplier

    Returns:
        WAV audio bytes
    """
    from kokoro import KPipeline
    import soundfile as sf
    import numpy as np

    pipeline = KPipeline(lang_code=lang_code)

    audio_chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        audio_chunks.append(audio)

    if not audio_chunks:
        raise ValueError("No audio generated")

    full_audio = np.concatenate(audio_chunks)

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, full_audio, 24000, format='WAV')
    wav_buffer.seek(0)
    return wav_buffer.read()

def get_kokoro_voices() -> list:
    """Get available Kokoro voices."""
    return [
        {"id": "af_heart", "name": "Heart", "gender": "female", "lang": "a"},
        {"id": "af_bella", "name": "Bella", "gender": "female", "lang": "a"},
        {"id": "af_sarah", "name": "Sarah", "gender": "female", "lang": "a"},
        {"id": "af_nicole", "name": "Nicole", "gender": "female", "lang": "a"},
        {"id": "af_sky", "name": "Sky", "gender": "female", "lang": "a"},
        {"id": "am_adam", "name": "Adam", "gender": "male", "lang": "a"},
        {"id": "am_michael", "name": "Michael", "gender": "male", "lang": "a"},
        {"id": "bf_emma", "name": "Emma", "gender": "female", "lang": "b"},
        {"id": "bf_isabella", "name": "Isabella", "gender": "female", "lang": "b"},
        {"id": "bm_george", "name": "George", "gender": "male", "lang": "b"},
        {"id": "bm_lewis", "name": "Lewis", "gender": "male", "lang": "b"},
    ]

def tts_elevenlabs(
    text: str,
    api_key: Optional[str] = None,
    voice_id: str = 'JBFqnCBsd6RMkjVDRZzb',
    model_id: str = 'eleven_multilingual_v2',
    output_format: str = 'mp3_44100_128'
) -> bytes:
    """
    Generate speech using ElevenLabs API.

    Returns:
        MP3 audio bytes
    """
    if api_key is None:
        api_key = os.environ.get('ELEVENLABS_API_KEY')

    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not set")

    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=api_key)

    audio_generator = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format
    )

    return b''.join(chunk for chunk in audio_generator)

async def tts_elevenlabs_stream(
    text: str,
    api_key: Optional[str] = None,
    voice_id: str = 'JBFqnCBsd6RMkjVDRZzb',
    model_id: str = 'eleven_turbo_v2_5',
    on_chunk: Optional[Callable[[bytes], None]] = None
) -> bytes:
    """
    Stream TTS via ElevenLabs WebSocket for lowest latency.

    Args:
        text: Text to synthesize
        api_key: ElevenLabs API key
        voice_id: Voice to use
        model_id: Model (eleven_turbo_v2_5 for fastest)
        on_chunk: Callback for each audio chunk

    Returns:
        Complete audio bytes
    """
    import websockets

    if api_key is None:
        api_key = os.environ.get('ELEVENLABS_API_KEY')

    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}"

    all_audio = []

    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            "xi_api_key": api_key
        }))

        await ws.send(json.dumps({"text": text}))
        await ws.send(json.dumps({"text": ""}))

        async for message in ws:
            data = json.loads(message)
            if "audio" in data:
                chunk = base64.b64decode(data["audio"])
                all_audio.append(chunk)
                if on_chunk:
                    on_chunk(chunk)
            if data.get("isFinal"):
                break

    return b''.join(all_audio)

def get_elevenlabs_voices(api_key: Optional[str] = None) -> list:
    """Get available ElevenLabs voices."""
    if api_key is None:
        api_key = os.environ.get('ELEVENLABS_API_KEY')

    if not api_key:
        return []

    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=api_key)
        voices = client.voices.get_all()
        return [{"id": v.voice_id, "name": v.name} for v in voices.voices]
    except Exception:
        return []

async def openai_realtime_connect(
    api_key: Optional[str] = None,
    model: str = "gpt-4o-realtime-preview-2024-12-17",
    voice: str = "alloy",
    instructions: str = "You are a helpful assistant."
):
    """
    Connect to OpenAI Realtime API.

    Returns:
        WebSocket connection
    """
    import websockets

    api_key = api_key or os.environ.get('OPENAI_API_KEY')

    url = f"wss://api.openai.com/v1/realtime?model={model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1"
    }

    ws = await websockets.connect(url, extra_headers=headers)

    await ws.send(json.dumps({
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "instructions": instructions,
            "voice": voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper-1"},
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500
            }
        }
    }))

    while True:
        msg = await ws.recv()
        event = json.loads(msg)
        if event.get("type") == "session.created":
            break
        elif event.get("type") == "error":
            await ws.close()
            raise Exception(f"OpenAI Realtime error: {event}")

    return ws

async def openai_realtime_send_audio(ws, audio_data: bytes):
    """Send audio to OpenAI Realtime (PCM16, 24kHz, mono)."""
    await ws.send(json.dumps({
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(audio_data).decode()
    }))

async def openai_realtime_send_text(ws, text: str):
    """Send text message to OpenAI Realtime."""
    await ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}]
        }
    }))
    await ws.send(json.dumps({"type": "response.create"}))

async def openai_realtime_receive(ws, on_audio=None, on_text=None):
    """
    Receive response from OpenAI Realtime.

    Args:
        ws: WebSocket connection
        on_audio: Callback for audio chunks (bytes)
        on_text: Callback for text chunks (str)

    Returns:
        Tuple of (full_audio_bytes, full_text)
    """
    audio_chunks = []
    text_chunks = []

    async for message in ws:
        event = json.loads(message)
        event_type = event.get("type", "")

        if event_type == "response.audio.delta":
            audio = base64.b64decode(event.get("delta", ""))
            audio_chunks.append(audio)
            if on_audio:
                on_audio(audio)

        elif event_type == "response.text.delta":
            text = event.get("delta", "")
            text_chunks.append(text)
            if on_text:
                on_text(text)

        elif event_type == "response.done":
            break

    return b''.join(audio_chunks), ''.join(text_chunks)

async def tts_openai_realtime(
    text: str,
    api_key: Optional[str] = None,
    voice: str = "alloy",
    on_chunk: Optional[Callable[[bytes], None]] = None
) -> bytes:
    """
    Use OpenAI Realtime API for TTS.

    Returns PCM16 audio at 24kHz.
    """
    ws = await openai_realtime_connect(api_key=api_key, voice=voice)
    try:
        await openai_realtime_send_text(ws, f"Please repeat exactly: {text}")
        audio, _ = await openai_realtime_receive(ws, on_audio=on_chunk)
        return audio
    finally:
        await ws.close()

def get_openai_voices() -> list:
    """Get available OpenAI Realtime voices."""
    return [
        {"id": "alloy", "name": "Alloy"},
        {"id": "echo", "name": "Echo"},
        {"id": "shimmer", "name": "Shimmer"},
        {"id": "ash", "name": "Ash"},
        {"id": "ballad", "name": "Ballad"},
        {"id": "coral", "name": "Coral"},
        {"id": "sage", "name": "Sage"},
        {"id": "verse", "name": "Verse"},
    ]

async def gemini_live_connect(
    api_key: Optional[str] = None,
    model: str = "gemini-2.0-flash-exp",
    voice: str = "Puck",
    system_instruction: str = "You are a helpful assistant."
):
    """
    Connect to Gemini Live API.

    Returns:
        WebSocket connection
    """
    import websockets

    api_key = api_key or os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')

    url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={api_key}"

    ws = await websockets.connect(url)

    await ws.send(json.dumps({
        "setup": {
            "model": f"models/{model}",
            "generation_config": {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {"voice_name": voice}
                    }
                }
            },
            "system_instruction": {"parts": [{"text": system_instruction}]}
        }
    }))

    response = await ws.recv()
    data = json.loads(response)
    if "setupComplete" not in data:
        await ws.close()
        raise Exception(f"Gemini Live setup failed: {data}")

    return ws

async def gemini_live_send_audio(ws, audio_data: bytes, mime_type: str = "audio/pcm"):
    """Send audio to Gemini Live."""
    await ws.send(json.dumps({
        "realtime_input": {
            "media_chunks": [{
                "data": base64.b64encode(audio_data).decode(),
                "mime_type": mime_type
            }]
        }
    }))

async def gemini_live_send_text(ws, text: str):
    """Send text message to Gemini Live."""
    await ws.send(json.dumps({
        "client_content": {
            "turns": [{"role": "user", "parts": [{"text": text}]}],
            "turn_complete": True
        }
    }))

async def gemini_live_receive(ws, on_audio=None, on_text=None):
    """
    Receive response from Gemini Live.

    Returns:
        Tuple of (full_audio_bytes, full_text)
    """
    audio_chunks = []
    text_chunks = []

    async for message in ws:
        data = json.loads(message)

        if "serverContent" in data:
            content = data["serverContent"]

            if "modelTurn" in content:
                for part in content["modelTurn"].get("parts", []):
                    if "inlineData" in part:
                        audio = base64.b64decode(part["inlineData"].get("data", ""))
                        audio_chunks.append(audio)
                        if on_audio:
                            on_audio(audio)
                    elif "text" in part:
                        text_chunks.append(part["text"])
                        if on_text:
                            on_text(part["text"])

            if content.get("turnComplete"):
                break

    return b''.join(audio_chunks), ''.join(text_chunks)

async def tts_gemini_live(
    text: str,
    api_key: Optional[str] = None,
    voice: str = "Puck",
    on_chunk: Optional[Callable[[bytes], None]] = None
) -> bytes:
    """
    Use Gemini Live API for TTS.

    Returns PCM audio.
    """
    ws = await gemini_live_connect(api_key=api_key, voice=voice)
    try:
        await gemini_live_send_text(ws, f"Please repeat exactly: {text}")
        audio, _ = await gemini_live_receive(ws, on_audio=on_chunk)
        return audio
    finally:
        await ws.close()

def get_gemini_voices() -> list:
    """Get available Gemini Live voices."""
    return [
        {"id": "Puck", "name": "Puck"},
        {"id": "Charon", "name": "Charon"},
        {"id": "Kore", "name": "Kore"},
        {"id": "Fenrir", "name": "Fenrir"},
        {"id": "Aoede", "name": "Aoede"},
    ]

_qwen3_model_cache = {}

def _get_qwen3_model(
    model_size: str = "1.7B",
    model_type: str = "custom_voice",
    device: str = "auto",
):
    """Load and cache a Qwen3-TTS model."""
    cache_key = (model_size, model_type, device)
    if cache_key in _qwen3_model_cache:
        return _qwen3_model_cache[cache_key]

    import torch
    from huggingface_hub import snapshot_download

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    size_tag = "0.6B" if "0.6" in model_size else "1.7B"
    type_map = {
        "custom_voice": f"Qwen/Qwen3-TTS-12Hz-{size_tag}-CustomVoice",
        "base": f"Qwen/Qwen3-TTS-12Hz-{size_tag}-Base",
        "voice_design": f"Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    }

    repo_id = type_map.get(model_type, type_map["custom_voice"])

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "qwen-tts")
    model_dir = os.path.join(cache_dir, repo_id.split("/")[-1])

    if not os.path.exists(os.path.join(model_dir, "config.json")):
        os.makedirs(cache_dir, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=model_dir)

    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        raise ImportError("qwen_tts package not found. Install from: https://github.com/QwenLM/Qwen3-TTS or pip install qwen-tts")

    model = Qwen3TTSModel.from_pretrained(
        model_dir, device_map=device, dtype=dtype
    )

    _qwen3_model_cache.clear()
    _qwen3_model_cache[cache_key] = model
    return model

def tts_qwen3(
    text: str,
    voice: str = "ryan",
    language: str = "auto",
    model_size: str = "1.7B",
    device: str = "auto",
    speed: float = 1.0,
    ref_audio: str = None,
    ref_text: str = None,
    instruct: str = None,
) -> bytes:
    """
    Generate speech using Qwen3-TTS local model.

    Supports three modes based on arguments:
    - Custom voice (default): Use a preset speaker name
    - Voice clone: Provide ref_audio (path) to clone a voice
    - Voice design: Provide instruct (text description) to design a voice

    Args:
        text: Text to synthesize
        voice: Speaker name for custom voice mode
            (aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian)
        language: Language (auto, chinese, english, japanese, korean, french, etc.)
        model_size: '0.6B' or '1.7B'
        device: 'auto', 'cuda', 'mps', 'cpu'
        speed: Speech speed (not directly supported, reserved)
        ref_audio: Path to reference audio for voice cloning
        ref_text: Transcript of reference audio (recommended for cloning)
        instruct: Natural language voice description for voice design mode

    Returns:
        WAV audio bytes
    """
    import numpy as np
    import soundfile as sf

    if ref_audio:
        model = _get_qwen3_model(model_size, "base", device)
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
    elif instruct:
        model = _get_qwen3_model(model_size, "voice_design", device)
        wavs, sr = model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
        )
    else:
        model = _get_qwen3_model(model_size, "custom_voice", device)
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=voice.lower().replace(" ", "_"),
        )

    if not wavs:
        raise ValueError("Qwen3-TTS generated no audio")

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, wavs[0], sr, format='WAV')
    wav_buffer.seek(0)
    return wav_buffer.read()

def get_qwen3_voices() -> list:
    """Get available Qwen3-TTS preset voices."""
    return [
        {"id": "aiden", "name": "Aiden", "gender": "male"},
        {"id": "dylan", "name": "Dylan", "gender": "male"},
        {"id": "eric", "name": "Eric", "gender": "male"},
        {"id": "ryan", "name": "Ryan", "gender": "male"},
        {"id": "serena", "name": "Serena", "gender": "female"},
        {"id": "vivian", "name": "Vivian", "gender": "female"},
        {"id": "sohee", "name": "Sohee", "gender": "female"},
        {"id": "ono_anna", "name": "Ono Anna", "gender": "female"},
        {"id": "uncle_fu", "name": "Uncle Fu", "gender": "male"},
    ]

def tts_gtts(text: str, lang: str = "en") -> bytes:
    """
    Generate speech using gTTS.

    Returns MP3 audio bytes.
    """
    from gtts import gTTS

    tts = gTTS(text=text, lang=lang)

    mp3_buffer = io.BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)
    return mp3_buffer.read()

def get_gtts_voices() -> list:
    """Get available gTTS languages."""
    return [
        {"id": "en", "name": "English"},
        {"id": "es", "name": "Spanish"},
        {"id": "fr", "name": "French"},
        {"id": "de", "name": "German"},
        {"id": "it", "name": "Italian"},
        {"id": "pt", "name": "Portuguese"},
        {"id": "ja", "name": "Japanese"},
        {"id": "ko", "name": "Korean"},
        {"id": "zh-CN", "name": "Chinese"},
    ]

def text_to_speech(
    text: str,
    engine: str = "kokoro",
    voice: Optional[str] = None,
    **kwargs
) -> bytes:
    """
    Unified TTS interface.

    Args:
        text: Text to synthesize
        engine: TTS engine (kokoro, qwen3, elevenlabs, openai, gemini, gtts)
        voice: Voice ID (engine-specific)
        **kwargs: Engine-specific options

    Returns:
        Audio bytes (format depends on engine)
    """
    engine = engine.lower()

    if engine == "kokoro":
        voice = voice or "af_heart"
        voices = {v["id"]: v for v in get_kokoro_voices()}
        lang_code = voices.get(voice, {}).get("lang", "a")
        return tts_kokoro(text, voice=voice, lang_code=lang_code, **kwargs)

    elif engine in ("qwen3", "qwen3-tts", "qwen"):
        voice = voice or "ryan"
        return tts_qwen3(text, voice=voice, **kwargs)

    elif engine == "elevenlabs":
        voice = voice or "JBFqnCBsd6RMkjVDRZzb"
        return tts_elevenlabs(text, voice_id=voice, **kwargs)

    elif engine == "openai":
        voice = voice or "alloy"
        return asyncio.run(tts_openai_realtime(text, voice=voice, **kwargs))

    elif engine == "gemini":
        voice = voice or "Puck"
        return asyncio.run(tts_gemini_live(text, voice=voice, **kwargs))

    elif engine == "gtts":
        lang = voice if voice and len(voice) <= 5 else "en"
        return tts_gtts(text, lang=lang)

    else:
        raise ValueError(f"Unknown TTS engine: {engine}")

def get_available_voices(engine: str = "kokoro") -> list:
    """Get available voices for an engine."""
    engine = engine.lower()

    if engine == "kokoro":
        return get_kokoro_voices()
    elif engine in ("qwen3", "qwen3-tts", "qwen"):
        return get_qwen3_voices()
    elif engine == "elevenlabs":
        return get_elevenlabs_voices()
    elif engine == "openai":
        return get_openai_voices()
    elif engine == "gemini":
        return get_gemini_voices()
    elif engine == "gtts":
        return get_gtts_voices()
    else:
        return []

def get_available_engines() -> dict:
    """Get info about available TTS engines."""
    engines = {
        "kokoro": {
            "name": "Kokoro",
            "type": "local",
            "available": False,
            "description": "Local neural TTS (82M params)",
            "install": "pip install kokoro soundfile"
        },
        "qwen3": {
            "name": "Qwen3-TTS",
            "type": "local",
            "available": False,
            "description": "Local high-quality multilingual TTS (0.6B/1.7B)",
            "install": "pip install qwen-tts torch torchaudio transformers"
        },
        "elevenlabs": {
            "name": "ElevenLabs",
            "type": "cloud",
            "available": False,
            "description": "High-quality cloud TTS",
            "requires": "ELEVENLABS_API_KEY"
        },
        "openai": {
            "name": "OpenAI Realtime",
            "type": "cloud",
            "available": False,
            "description": "OpenAI real-time voice API",
            "requires": "OPENAI_API_KEY"
        },
        "gemini": {
            "name": "Gemini Live",
            "type": "cloud",
            "available": False,
            "description": "Google Gemini real-time voice",
            "requires": "GOOGLE_API_KEY or GEMINI_API_KEY"
        },
        "gtts": {
            "name": "Google TTS",
            "type": "cloud",
            "available": False,
            "description": "Free Google TTS (fallback)"
        }
    }

    try:
        from kokoro import KPipeline
        engines["kokoro"]["available"] = True
    except ImportError:
        pass

    try:
        from qwen_tts import Qwen3TTSModel
        engines["qwen3"]["available"] = True
    except ImportError:
        pass

    if os.environ.get('ELEVENLABS_API_KEY'):
        engines["elevenlabs"]["available"] = True

    if os.environ.get('OPENAI_API_KEY'):
        engines["openai"]["available"] = True

    if os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY'):
        engines["gemini"]["available"] = True

    try:
        from gtts import gTTS
        engines["gtts"]["available"] = True
    except ImportError:
        pass

    return engines

def pcm16_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """Convert raw PCM16 audio to WAV format."""
    import struct

    wav_buffer = io.BytesIO()
    wav_buffer.write(b'RIFF')
    wav_buffer.write(struct.pack('<I', 36 + len(pcm_data)))
    wav_buffer.write(b'WAVE')
    wav_buffer.write(b'fmt ')
    wav_buffer.write(struct.pack('<I', 16))
    wav_buffer.write(struct.pack('<H', 1))
    wav_buffer.write(struct.pack('<H', channels))
    wav_buffer.write(struct.pack('<I', sample_rate))
    wav_buffer.write(struct.pack('<I', sample_rate * channels * 2))
    wav_buffer.write(struct.pack('<H', channels * 2))
    wav_buffer.write(struct.pack('<H', 16))
    wav_buffer.write(b'data')
    wav_buffer.write(struct.pack('<I', len(pcm_data)))
    wav_buffer.write(pcm_data)

    wav_buffer.seek(0)
    return wav_buffer.read()

def wav_to_pcm16(wav_data: bytes) -> tuple:
    """Extract PCM16 data from WAV. Returns (pcm_data, sample_rate)."""
    import struct

    if wav_data[:4] != b'RIFF' or wav_data[8:12] != b'WAVE':
        raise ValueError("Invalid WAV data")

    pos = 12
    sample_rate = 24000
    while pos < len(wav_data) - 8:
        chunk_id = wav_data[pos:pos+4]
        chunk_size = struct.unpack('<I', wav_data[pos+4:pos+8])[0]

        if chunk_id == b'fmt ':
            sample_rate = struct.unpack('<I', wav_data[pos+12:pos+16])[0]
        elif chunk_id == b'data':
            return wav_data[pos+8:pos+8+chunk_size], sample_rate

        pos += 8 + chunk_size

    raise ValueError("No data chunk found in WAV")

def audio_to_base64(audio_data: bytes) -> str:
    """Encode audio to base64 string."""
    return base64.b64encode(audio_data).decode('utf-8')

def base64_to_audio(b64_string: str) -> bytes:
    """Decode base64 string to audio bytes."""
    return base64.b64decode(b64_string)


# ─── Music generation ─────────────────────────────────────────────────

def music_musicgen_mlx(
    prompt: str,
    duration: int = 10,
    model: str = "facebook/musicgen-medium",
) -> bytes:
    """Generate music natively on Apple Silicon via MLX.

    Uses Apple's mlx-examples MusicGen port (vendored into npcpy.gen.mlx_musicgen).
    Runs on the Apple GPU via MLX — no CUDA/MPS/CPU fallback needed.
    Returns WAV bytes at 32 kHz.
    """
    import numpy as np
    import scipy.io.wavfile as wavfile
    import mlx.core as mx
    from npcpy.gen.mlx_musicgen.musicgen import MusicGen

    mg = MusicGen.from_pretrained(model)
    # MusicGen's MLX port uses ~50 tokens/sec; max_steps maps linearly to duration
    max_steps = max(50, int(duration * 50))
    audio = mg.generate(prompt, max_steps=max_steps)
    # audio is mx.array in [-1, 1]; convert to int16 WAV
    arr = np.array(mx.clip(audio, -1, 1))
    pcm = (arr * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, mg.sampling_rate, pcm)
    return buf.getvalue()


def music_musicgen_local(
    prompt: str,
    duration: int = 10,
    model: str = "facebook/musicgen-small",
) -> bytes:
    """Generate music locally with Meta MusicGen via transformers.

    Returns WAV bytes (MusicGen outputs at 32 kHz).
    """
    import numpy as np
    import scipy.io.wavfile as wavfile
    import torch
    # Use concrete classes to sidestep AutoProcessor's auto-discovery path.
    from transformers import MusicgenProcessor, MusicgenForConditionalGeneration

    processor = MusicgenProcessor.from_pretrained(model)
    mg = MusicgenForConditionalGeneration.from_pretrained(model)

    # MusicGen is numerically unstable on MPS (produces NaN/inf probs during
    # sampling) — stick to CUDA or CPU.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    mg = mg.to(device)
    mg.eval()

    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

    # MusicGen uses ~50 tokens/sec at 32 kHz
    max_new_tokens = int(duration * 50)

    with torch.no_grad():
        audio_values = mg.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, guidance_scale=3.0)

    sample_rate = mg.config.audio_encoder.sampling_rate
    arr = audio_values[0, 0].detach().cpu().numpy()
    # normalize to int16
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, pcm)
    return buf.getvalue()


def music_replicate(
    prompt: str,
    duration: int = 10,
    model: str = "meta/musicgen",
    version: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bytes:
    """Generate music via Replicate. Works with musicgen, stable-audio, riffusion."""
    import requests

    api_key = api_key or os.environ.get("REPLICATE_API_TOKEN") or os.environ.get("REPLICATE_API_KEY")
    if not api_key:
        raise RuntimeError("REPLICATE_API_TOKEN env var required for Replicate music generation")

    # Resolve model → latest version automatically if not given.
    if not version:
        owner_model = model
        r = requests.get(
            f"https://api.replicate.com/v1/models/{owner_model}",
            headers={"Authorization": f"Token {api_key}"},
            timeout=30,
        )
        r.raise_for_status()
        version = r.json()["latest_version"]["id"]

    inputs: dict = {"prompt": prompt, "duration": duration}
    if "stable-audio" in model:
        inputs = {"prompt": prompt, "seconds_total": duration}
    elif "riffusion" in model:
        inputs = {"prompt_a": prompt}

    create = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
        json={"version": version, "input": inputs},
        timeout=30,
    )
    create.raise_for_status()
    pred = create.json()
    poll_url = pred["urls"]["get"]

    import time
    for _ in range(180):  # up to ~6 minutes
        p = requests.get(poll_url, headers={"Authorization": f"Token {api_key}"}, timeout=30).json()
        status = p.get("status")
        if status == "succeeded":
            out = p.get("output")
            audio_url = out[0] if isinstance(out, list) else out
            return requests.get(audio_url, timeout=120).content
        if status in ("failed", "canceled"):
            raise RuntimeError(f"Replicate prediction {status}: {p.get('error')}")
        time.sleep(2)
    raise TimeoutError("Replicate music generation timed out")


def music_elevenlabs_sfx(
    prompt: str,
    duration: float = 10.0,
    api_key: Optional[str] = None,
) -> bytes:
    """Generate a short sound effect / music clip via ElevenLabs sound-generation API."""
    import requests

    api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY env var required for ElevenLabs sound generation")

    # ElevenLabs caps sound-generation at 22 seconds.
    duration = max(0.5, min(22.0, float(duration)))

    r = requests.post(
        "https://api.elevenlabs.io/v1/sound-generation",
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={
            "text": prompt,
            "duration_seconds": duration,
            "prompt_influence": 0.3,
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.content  # MP3 bytes


def _music_one(provider: str, prompt: str, model: Optional[str], duration: int, api_key: Optional[str]) -> dict:
    p = (provider or "").lower()
    if p in ("local", "musicgen", "transformers", "meta", "mlx", "apple"):
        # Pick the best local backend for this machine: MLX on Apple Silicon,
        # CUDA/CPU via transformers elsewhere.
        try:
            import mlx.core as _mx  # noqa: F401
            m = model or "facebook/musicgen-medium"
            return {"audio": music_musicgen_mlx(prompt, duration=duration, model=m),
                    "format": "wav", "provider": "musicgen-local-mlx", "model": m}
        except ImportError:
            m = model or "facebook/musicgen-small"
            return {"audio": music_musicgen_local(prompt, duration=duration, model=m),
                    "format": "wav", "provider": "musicgen-local-torch", "model": m}
    if p in ("replicate",):
        m = model or "meta/musicgen"
        return {"audio": music_replicate(prompt, duration=duration, model=m, api_key=api_key),
                "format": "wav", "provider": "replicate", "model": m}
    if p in ("elevenlabs", "stability", "11labs"):
        return {"audio": music_elevenlabs_sfx(prompt, duration=duration, api_key=api_key),
                "format": "mp3", "provider": "elevenlabs", "model": "sound-generation"}
    raise ValueError(f"Unknown music provider: {provider!r}")


def generate_music(
    prompt: str,
    provider: str = "replicate",
    model: Optional[str] = None,
    duration: int = 10,
    api_key: Optional[str] = None,
) -> dict:
    """Unified music-generation entry point with automatic provider fallback.

    Tries the requested provider first; on failure, walks the other providers
    that have credentials available so the user always gets audio back.
    """
    requested = (provider or "local").lower()
    order = [requested]
    for cand in ("local", "replicate", "elevenlabs"):
        if cand not in order:
            order.append(cand)

    errors: list[str] = []
    for prov in order:
        # Skip cloud providers that have no credentials configured.
        if prov == "replicate" and not (api_key or os.environ.get("REPLICATE_API_TOKEN") or os.environ.get("REPLICATE_API_KEY")):
            errors.append(f"{prov}: no REPLICATE_API_TOKEN")
            continue
        if prov in ("elevenlabs", "11labs", "stability") and not (api_key or os.environ.get("ELEVENLABS_API_KEY")):
            errors.append(f"{prov}: no ELEVENLABS_API_KEY")
            continue
        try:
            return _music_one(prov, prompt, model if prov == requested else None, duration, api_key)
        except Exception as e:
            errors.append(f"{prov}: {e}")
            continue

    raise RuntimeError("All music providers failed:\n  - " + "\n  - ".join(errors))
