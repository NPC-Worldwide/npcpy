def generate_video_minimax(
    prompt: str,
    model: str = "MiniMax-H3",
    output_path: str = "",
    api_key: str = "",
    base_url: str = "https://api.minimax.io",
    resolution: str = "2K",
    duration: int = 6,
    ratio: str = "adaptive",
    poll_interval: float = 5.0,
    timeout: float = 900.0,
):
    """Generate a MiniMax H3 video and wait for its asynchronous result."""
    import os
    import time
    import requests

    token = api_key or os.environ.get("MINIMAX_API_KEY") or os.environ.get("MINIMAX_API_TOKEN")
    if not token:
        raise ValueError("A MiniMax API key is required")
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not 4 <= int(duration) <= 15:
        raise ValueError("duration must be between 4 and 15 seconds")

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "content": [{"type": "text", "text": prompt}],
        "resolution": resolution,
        "duration": int(duration),
        "ratio": ratio,
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/v2/video_generation",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    task_id = response.json().get("task_id")
    if not task_id:
        raise RuntimeError("MiniMax video generation response did not include task_id")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status_response = requests.get(
            f"{base_url.rstrip('/')}/v2/query/video_generation/{task_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
        status_response.raise_for_status()
        task = status_response.json().get("task", status_response.json())
        status = str(task.get("status", "")).lower()
        if status in {"success", "succeeded", "completed", "finished"}:
            content = task.get("content") or {}
            video_url = content.get("url") or task.get("video_url")
            if not video_url:
                raise RuntimeError("MiniMax completed task did not include a video URL")
            if not output_path:
                from npcpy.npc_sysenv import get_videos_dir
                os.makedirs(get_videos_dir(), exist_ok=True)
                output_path = os.path.join(get_videos_dir(), "minimax_h3.mp4")
            video_response = requests.get(video_url, timeout=120)
            video_response.raise_for_status()
            with open(output_path, "wb") as output_file:
                output_file.write(video_response.content)
            return output_path
        if status in {"failed", "failure", "error", "cancelled", "canceled"}:
            raise RuntimeError(task.get("error") or f"MiniMax video generation failed with status {status}")
        time.sleep(poll_interval)
    raise TimeoutError("Timed out waiting for MiniMax video generation")


def generate_video_diffusers(
    prompt,
    model,
    npc=None,
    device="gpu",
    output_path="",
    num_inference_steps=5,
    num_frames=25,
    height=256,
    width=256,
):

    import torch
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    import numpy as np
    import os 
    import cv2

    
    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float32
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    output = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    def save_frames_to_video(frames, output_path, fps=8):
        """Handle the specific 5D array format (1, num_frames, H, W, 3) with proper type conversion"""
        
        if not (
            isinstance(frames, np.ndarray)
            and frames.ndim == 5
            and frames.shape[-1] == 3
        ):
            raise ValueError(
                f"Unexpected frame format. Expected 5D RGB array, got {frames.shape}"
            )

        
        frames = (frames[0] * 255).astype(np.uint8)  

        
        height, width = frames.shape[1:3]

        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not video_writer.isOpened():
            raise IOError(f"Could not open video writer for {output_path}")

        
        for frame in frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video_writer.release()
        print(f"Successfully saved {frames.shape[0]} frames to {output_path}")

    from npcpy.npc_sysenv import get_videos_dir
    _vid_dir = get_videos_dir()
    os.makedirs(_vid_dir, exist_ok=True)
    if output_path == "":
        output_path = os.path.join(_vid_dir, prompt[0:8] + ".mp4")
    save_frames_to_video(output.frames, output_path)
    return output_path

def generate_video_veo3(
    prompt: str,
    model: str, 
    negative_prompt: str = "",
    output_path: str = "",
):
    """
    Generate video using Google's Veo 3 API with synchronized audio.
    """
    import time
    import os
    from google import genai
    from google.genai import types
    api_key =os.environ.get('GEMINI_API_KEY')

    client = genai.Client(        api_key=api_key)
    
    config = types.GenerateVideosConfig()
    if negative_prompt:
        config.negative_prompt = negative_prompt
    
    operation = client.models.generate_videos(
        model=model,
        prompt=prompt,
        config=config,
    )
    
    while not operation.done:
        time.sleep(20)
        operation = client.operations.get(operation)
    
    generated_video = operation.result.generated_videos[0]
    
    from npcpy.npc_sysenv import get_videos_dir
    videos_dir = get_videos_dir()
    os.makedirs(videos_dir, exist_ok=True)
    if output_path == "":
        safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        output_path = os.path.join(videos_dir, safe_prompt.replace(" ", "_") + "_veo3.mp4")
    
    client.files.download(file=generated_video.video)
    generated_video.video.save(output_path)
    
    return output_path
