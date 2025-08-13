import time
import os

def generate_video_genai(
    prompt,
    client=None,
    model="veo-3.0-generate-preview",
    output_path="",
    wait_interval=10,
    max_wait_time=300  # 5 minutes maximum wait time
):
    """
    Generate a video using Google's Generative AI Veo model.
    
    Args:
        prompt (str): Text prompt describing the video to generate
        client (genai.Client, optional): Initialized Google Generative AI client. 
                                         If None, will attempt to create one.
        model (str, optional): Video generation model to use. 
                               Defaults to "veo-3.0-generate-preview"
        output_path (str, optional): Path to save the generated video. 
                                     If empty, will use default directory
        wait_interval (int, optional): Seconds between polling for job completion. 
                                       Defaults to 10
        max_wait_time (int, optional): Maximum total wait time in seconds. 
                                       Defaults to 300 (5 minutes)
    
    Returns:
        str: Path to the generated video file
    """
    # Import here to allow optional dependency
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Google Generative AI package is required. "
                          "Install with 'pip install google-generativeai'")
    
    # Create client if not provided
    if client is None:
        client = genai.Client()
    
    # Ensure output directory exists
    os.makedirs("~/.npcsh/videos/", exist_ok=True)
    
    # Set default output path if not provided
    if not output_path:
        output_path = os.path.expanduser(f"~/.npcsh/videos/{prompt[:8]}_genai.mp4")
    
    # Start the generation job
    operation = client.models.generate_videos(
        model=model,
        prompt=prompt,
    )
    
    # Track waiting time
    total_wait_time = 0
    
    # Poll for the result
    while not operation.done and total_wait_time < max_wait_time:
        print("Waiting for video generation to complete...")
        time.sleep(wait_interval)
        total_wait_time += wait_interval
        operation = client.operations.get(operation)
    
    # Check if generation completed or timed out
    if not operation.done:
        raise TimeoutError(f"Video generation timed out after {max_wait_time} seconds")
    
    # Download the final video
    video = operation.response.generated_videos[0]
    video.video.save(output_path)
    
    print(f"Generated video saved to {output_path}")
    return output_path
