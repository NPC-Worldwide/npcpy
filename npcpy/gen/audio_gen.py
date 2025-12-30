"""
Audio Generation Module for NPC
Supports multiple TTS engines including real-time voice APIs.

TTS Engines:
- Kokoro: Local neural TTS (default)
- ElevenLabs: Cloud TTS with streaming
- OpenAI: Realtime voice API
- Gemini: Live API for real-time voice
- gTTS: Google TTS fallback

Usage:
    from npcpy.gen.audio_gen import text_to_speech

    audio = text_to_speech("Hello world", engine="kokoro", voice="af_heart")

For STT, see npcpy.data.audio
"""

import os
import io
import base64
import json
import asyncio
import tempfile
from typing import Optional, Callable, Any


# =============================================================================
# Kokoro TTS (Local Neural)
# =============================================================================

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


# =============================================================================
# ElevenLabs TTS
# =============================================================================

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


# =============================================================================
# OpenAI Realtime Voice API
# =============================================================================

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


# =============================================================================
# Google Gemini Live API
# =============================================================================

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


# =============================================================================
# gTTS (Google Text-to-Speech) - Fallback
# =============================================================================

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


# =============================================================================
# Unified Interface
# =============================================================================

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
        engine: TTS engine (kokoro, elevenlabs, openai, gemini, gtts)
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
            "description": "Free Google TTS"
        }
    }

    try:
        from kokoro import KPipeline
        engines["kokoro"]["available"] = True
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


# =============================================================================
# Audio Utilities
# =============================================================================

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
