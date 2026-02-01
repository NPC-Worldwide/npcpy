import os
import numpy as np
import tempfile
import threading
import time
import queue
import re
import json
import subprocess
import logging

from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Audio constants
try:
    import pyaudio
    FORMAT = pyaudio.paInt16
except ImportError:
    FORMAT = 8  # paInt16 value fallback
CHANNELS = 1
RATE = 16000
CHUNK = 512


def convert_mp3_to_wav(mp3_file, wav_file):
    try:
        
        if os.path.exists(wav_file):
            os.remove(wav_file)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                mp3_file,
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "44100",
                wav_file,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error converting MP3 to WAV: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error during conversion: {e}")
        raise



def check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def audio_callback(in_data, frame_count, time_info, status):
    import pyaudio
    audio_queue = queue.Queue()
    audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)


def transcribe_recording(audio_data):
    if not audio_data:
        return None

    audio_np = (
        np.frombuffer(b"".join(audio_data), dtype=np.int16).astype(np.float32) / 32768.0
    )
    return run_transcription(audio_np)


def run_transcription(audio_np):
    try:
        temp_file = os.path.join(
            tempfile.gettempdir(), f"temp_recording_{int(time.time())}.wav"
        )
        with wave.open(temp_file, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes((audio_np * 32768).astype(np.int16).tobytes())
        whisper_model = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu")
        segments, info = whisper_model.transcribe(temp_file, language="en", beam_size=5)
        transcription = " ".join([segment.text for segment in segments])

        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            if temp_file not in cleanup_files:
                cleanup_files.append(temp_file)

        return transcription.strip()

    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return None


def transcribe_audio_file(file_path: str, language=None) -> str:
    """
    File-based transcription helper that prefers the local faster-whisper/whisper
    setup used elsewhere in this module.
    """
    # Try faster-whisper first
    try:
        from faster_whisper import WhisperModel  # type: ignore
        try:
            import torch  # type: ignore
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        model = WhisperModel("small", device=device)
        segments, _ = model.transcribe(file_path, language=language, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments if seg.text).strip()
        if text:
            return text
    except Exception:
        pass

    # Fallback to openai/whisper if available
    try:
        import whisper  # type: ignore
        model = whisper.load_model("small")
        result = model.transcribe(file_path, language=language)
        text = result.get("text", "").strip()
        if text:
            return text
    except Exception:
        pass

    return ""


# =============================================================================
# Speech-to-Text: Multi-Engine Support
# =============================================================================

def stt_whisper(
    audio_data: bytes,
    model_size: str = "base",
    language: str = None,
    device: str = "auto"
) -> dict:
    """
    Transcribe audio using local Whisper (faster-whisper).

    Args:
        audio_data: Audio bytes (WAV, MP3, etc.)
        model_size: Model size (tiny, base, small, medium, large-v3)
        language: Language code or None for auto-detect
        device: 'cpu', 'cuda', or 'auto'

    Returns:
        Dict with 'text', 'language', 'segments'
    """
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_data)
        temp_path = f.name

    try:
        segments, info = model.transcribe(
            temp_path,
            language=language,
            beam_size=5,
            vad_filter=True
        )

        segment_list = []
        text_parts = []
        for segment in segments:
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
            text_parts.append(segment.text)

        return {
            "text": " ".join(text_parts).strip(),
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": segment_list
        }
    finally:
        os.unlink(temp_path)


def stt_openai(
    audio_data: bytes,
    api_key: str = None,
    model: str = "whisper-1",
    language: str = None,
    response_format: str = "verbose_json",
    filename: str = "audio.wav"
) -> dict:
    """
    Transcribe audio using OpenAI Whisper API.

    Args:
        audio_data: Audio bytes
        api_key: OpenAI API key
        model: Model name (whisper-1)
        language: Optional language hint
        response_format: json, text, srt, verbose_json, vtt
        filename: Filename hint for format detection

    Returns:
        Dict with 'text', 'language', 'segments' (if verbose_json)
    """
    import requests

    api_key = api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}

    files = {"file": (filename, audio_data)}
    data = {"model": model, "response_format": response_format}
    if language:
        data["language"] = language

    response = requests.post(url, headers=headers, files=files, data=data)
    response.raise_for_status()

    if response_format == "verbose_json":
        result = response.json()
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", "en"),
            "duration": result.get("duration"),
            "segments": result.get("segments", [])
        }
    elif response_format == "json":
        return {"text": response.json().get("text", "").strip()}
    else:
        return {"text": response.text.strip()}


def stt_gemini(
    audio_data: bytes,
    api_key: str = None,
    model: str = "gemini-1.5-flash",
    language: str = None,
    mime_type: str = "audio/wav"
) -> dict:
    """
    Transcribe audio using Gemini API.

    Args:
        audio_data: Audio bytes
        api_key: Google/Gemini API key
        model: Gemini model
        language: Language hint
        mime_type: Audio MIME type

    Returns:
        Dict with 'text'
    """
    import google.generativeai as genai

    api_key = api_key or os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)

    prompt = "Transcribe this audio exactly. Output only the transcription, nothing else."
    if language:
        prompt = f"Transcribe this audio in {language}. Output only the transcription, nothing else."

    response = model_obj.generate_content([
        prompt,
        {"mime_type": mime_type, "data": audio_data}
    ])

    return {"text": response.text.strip()}


def stt_elevenlabs(
    audio_data: bytes,
    api_key: str = None,
    model_id: str = "scribe_v1",
    language: str = None
) -> dict:
    """
    Transcribe audio using ElevenLabs Scribe API.

    Args:
        audio_data: Audio bytes
        api_key: ElevenLabs API key
        model_id: Model (scribe_v1)
        language: Language code (ISO 639-1)

    Returns:
        Dict with 'text', 'language', 'words'
    """
    import requests

    api_key = api_key or os.environ.get('ELEVENLABS_API_KEY')
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not set")

    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": api_key}

    files = {"file": ("audio.wav", audio_data, "audio/wav")}
    data = {"model_id": model_id}
    if language:
        data["language_code"] = language

    response = requests.post(url, headers=headers, files=files, data=data)
    response.raise_for_status()

    result = response.json()
    return {
        "text": result.get("text", "").strip(),
        "language": result.get("language_code"),
        "words": result.get("words", [])
    }


def stt_groq(
    audio_data: bytes,
    api_key: str = None,
    model: str = "whisper-large-v3",
    language: str = None
) -> dict:
    """
    Transcribe audio using Groq's Whisper API (very fast).

    Args:
        audio_data: Audio bytes
        api_key: Groq API key
        model: whisper-large-v3 or whisper-large-v3-turbo
        language: Language code

    Returns:
        Dict with 'text'
    """
    import requests

    api_key = api_key or os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}

    files = {"file": ("audio.wav", audio_data, "audio/wav")}
    data = {"model": model}
    if language:
        data["language"] = language

    response = requests.post(url, headers=headers, files=files, data=data)
    response.raise_for_status()

    return {"text": response.json().get("text", "").strip()}


def speech_to_text(
    audio_data: bytes,
    engine: str = "whisper",
    language: str = None,
    **kwargs
) -> dict:
    """
    Unified STT interface.

    Args:
        audio_data: Audio bytes (WAV, MP3, etc.)
        engine: STT engine (whisper, openai, gemini, elevenlabs, groq)
        language: Language hint
        **kwargs: Engine-specific options

    Returns:
        Dict with at least 'text' key
    """
    engine = engine.lower()

    if engine == "whisper" or engine == "faster-whisper":
        try:
            return stt_whisper(audio_data, language=language, **kwargs)
        except ImportError:
            # Fallback to openai whisper
            import whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            try:
                model = whisper.load_model(kwargs.get("model_size", "base"))
                result = model.transcribe(temp_path, language=language)
                return {"text": result["text"].strip(), "language": result.get("language", "en")}
            finally:
                os.unlink(temp_path)

    elif engine == "openai":
        return stt_openai(audio_data, language=language, **kwargs)

    elif engine == "gemini":
        return stt_gemini(audio_data, language=language, **kwargs)

    elif engine == "elevenlabs":
        return stt_elevenlabs(audio_data, language=language, **kwargs)

    elif engine == "groq":
        return stt_groq(audio_data, language=language, **kwargs)

    else:
        raise ValueError(f"Unknown STT engine: {engine}")


def get_available_stt_engines() -> dict:
    """Get info about available STT engines."""
    engines = {
        "whisper": {
            "name": "Whisper (Local)",
            "type": "local",
            "available": False,
            "description": "OpenAI Whisper running locally",
            "install": "pip install faster-whisper"
        },
        "openai": {
            "name": "OpenAI Whisper API",
            "type": "cloud",
            "available": False,
            "description": "OpenAI's cloud Whisper API",
            "requires": "OPENAI_API_KEY"
        },
        "gemini": {
            "name": "Gemini",
            "type": "cloud",
            "available": False,
            "description": "Google Gemini transcription",
            "requires": "GOOGLE_API_KEY or GEMINI_API_KEY"
        },
        "elevenlabs": {
            "name": "ElevenLabs Scribe",
            "type": "cloud",
            "available": False,
            "description": "ElevenLabs speech-to-text",
            "requires": "ELEVENLABS_API_KEY"
        },
        "groq": {
            "name": "Groq Whisper",
            "type": "cloud",
            "available": False,
            "description": "Ultra-fast Whisper via Groq",
            "requires": "GROQ_API_KEY"
        }
    }

    # Check local whisper
    try:
        from faster_whisper import WhisperModel
        engines["whisper"]["available"] = True
    except ImportError:
        try:
            import whisper
            engines["whisper"]["available"] = True
        except ImportError:
            pass

    # Check API keys
    if os.environ.get('OPENAI_API_KEY'):
        engines["openai"]["available"] = True

    if os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY'):
        engines["gemini"]["available"] = True

    if os.environ.get('ELEVENLABS_API_KEY'):
        engines["elevenlabs"]["available"] = True

    if os.environ.get('GROQ_API_KEY'):
        engines["groq"]["available"] = True

    return engines




# =============================================================================
# TTS Playback Helpers (use unified audio_gen.text_to_speech)
# =============================================================================

def create_and_queue_audio(text, state, engine="kokoro", voice=None):
    """Create and play TTS audio using the unified engine interface.

    Args:
        text: Text to speak
        state: Dict with 'tts_is_speaking', 'tts_just_finished', 'running' keys
        engine: TTS engine name (kokoro, qwen3, elevenlabs, openai, gemini, gtts)
        voice: Voice ID (engine-specific)
    """
    import wave
    import uuid

    state["tts_is_speaking"] = True

    if not text.strip():
        state["tts_is_speaking"] = False
        return

    try:
        from npcpy.gen.audio_gen import text_to_speech

        audio_bytes = text_to_speech(text, engine=engine, voice=voice)

        # Write to temp file and play
        suffix = '.mp3' if engine in ('elevenlabs', 'gtts') else '.wav'
        tmp_path = os.path.join(tempfile.gettempdir(), f"npc_tts_{uuid.uuid4()}{suffix}")
        with open(tmp_path, 'wb') as f:
            f.write(audio_bytes)

        play_path = tmp_path
        if suffix == '.mp3':
            wav_path = tmp_path.replace('.mp3', '.wav')
            convert_mp3_to_wav(tmp_path, wav_path)
            play_path = wav_path

        play_audio(play_path, state)

        for p in set([tmp_path, play_path]):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    except Exception as e:
        logger.error(f"TTS error: {e}")
    finally:
        state["tts_is_speaking"] = False
        state["tts_just_finished"] = True


def play_audio(filename, state):
    """Play a WAV file via pyaudio with state awareness."""
    import pyaudio
    import wave

    PLAY_CHUNK = 4096

    wf = wave.open(filename, "rb")
    p = pyaudio.PyAudio()

    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
    )

    data = wf.readframes(PLAY_CHUNK)
    while data and state.get("running", True):
        stream.write(data)
        data = wf.readframes(PLAY_CHUNK)

    stream.stop_stream()
    stream.close()
    p.terminate()


def process_text_for_tts(text):
    """Clean text for TTS consumption."""
    text = re.sub(r"[*<>{}()\[\]&%#@^~`]", "", text)
    text = text.strip()
    text = re.sub(r"(\w)\.(\w)\.", r"\1 \2 ", text)
    text = re.sub(r"([.!?])(\w)", r"\1 \2", text)
    return text

