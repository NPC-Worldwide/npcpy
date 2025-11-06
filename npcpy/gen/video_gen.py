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

    os.makedirs("~/.npcsh/videos/", exist_ok=True)  
    if output_path == "":

        output_path = "~/.npcsh/videos/" + prompt[0:8] + ".mp4"
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
    
    os.makedirs(os.path.expanduser("~/.npcsh/videos/"), exist_ok=True)
    if output_path == "":
        safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        output_path = os.path.expanduser("~/.npcsh/videos/") + safe_prompt.replace(" ", "_") + "_veo3.mp4"
    
    client.files.download(file=generated_video.video)
    generated_video.video.save(output_path)
    
    return output_path