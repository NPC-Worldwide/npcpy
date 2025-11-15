
import os
import tempfile
import subprocess


def process_video(file_path, table_name):
    
    import cv2
    import base64

    embeddings = []
    texts = []
    try:
        video = cv2.VideoCapture(file_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(frame_count):
            ret, frame = video.read()
            if not ret:
                break

            
            n = 10  
        
        return embeddings, texts

    except Exception as e:
        print(f"Error processing video: {e}")
        return [], []  


def summarize_video_file(file_path: str, language: str = None, max_audio_seconds: int = 600) -> str:
    """
    Summarize a video using lightweight metadata plus optional audio transcript.
    Prefers the audio transcription helper in npcpy.data.audio when available.
    """
    meta_bits = []
    try:
        import cv2  # type: ignore

        video = cv2.VideoCapture(file_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps else 0
        meta_bits.append(
            f"Video file: {os.path.basename(file_path)} | {width}x{height} | {fps:.2f} fps | {frame_count} frames | ~{duration:.1f}s"
        )
        video.release()
    except Exception:
        meta_bits.append(f"Video file: {os.path.basename(file_path)}")

    # Extract audio track with ffmpeg if available
    audio_path = None
    try:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio.close()
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            file_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-t",
            str(max_audio_seconds),
            temp_audio.name,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio_path = temp_audio.name
    except Exception:
        audio_path = None

    transcript = ""
    if audio_path:
        try:
            try:
                from npcpy.data.audio import transcribe_audio_file
                transcript = transcribe_audio_file(audio_path, language=language)  # type: ignore
            except Exception:
                transcript = ""
        finally:
            try:
                os.remove(audio_path)
            except Exception:
                pass

    if transcript:
        meta_bits.append("Audio transcript:")
        meta_bits.append(transcript)
    else:
        meta_bits.append("[No transcript extracted; ensure ffmpeg and a transcription backend are installed]")

    return "\n".join(meta_bits)
