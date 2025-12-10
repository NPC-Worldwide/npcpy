"""
World model generation - predict next state(s) given current state and action.

Supports local world simulation models like:
- DIAMOND (diffusion world model for Atari)
- GameNGen (neural game engine)
- Dreamer/DreamerV3 (latent world models)

These are interactive models that take actions and maintain consistency,
unlike video gen which just produces video from prompts.
"""

from typing import List, Optional, Union, Dict, Any
import numpy as np


def world_step(
    frames: List[np.ndarray],
    action: Optional[Union[int, np.ndarray]] = None,
    model_path: str = None,
    model_type: str = "diamond",
    num_steps: int = 1,
    device: str = "cuda",
    **kwargs
) -> Dict[str, Any]:
    """
    Predict next frame(s) given frame history and optional action.

    Args:
        frames: List of recent frames as numpy arrays (H, W, C) or (C, H, W)
        action: Action to condition on. Int for discrete, array for continuous.
        model_path: Path to local model checkpoint
        model_type: One of "diamond", "gamengen", "dreamer"
        num_steps: Number of frames to predict
        device: "cuda" or "cpu"

    Returns:
        Dict with:
            - "frames": List of predicted frames as numpy arrays
            - "latent": Optional latent state for continuing simulation
            - "metadata": Model-specific info
    """

    if model_type == "diamond":
        return _step_diamond(frames, action, model_path, num_steps, device, **kwargs)
    elif model_type == "gamengen":
        return _step_gamengen(frames, action, model_path, num_steps, device, **kwargs)
    elif model_type == "dreamer":
        return _step_dreamer(frames, action, model_path, num_steps, device, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: diamond, gamengen, dreamer")


def world_rollout(
    initial_frames: List[np.ndarray],
    actions: List[Union[int, np.ndarray]],
    model_path: str = None,
    model_type: str = "diamond",
    device: str = "cuda",
    **kwargs
) -> Dict[str, Any]:
    """
    Roll out multiple steps given a sequence of actions.

    Args:
        initial_frames: Starting frame(s)
        actions: List of actions to execute in sequence
        model_path: Path to local model checkpoint
        model_type: One of "diamond", "gamengen", "dreamer"
        device: "cuda" or "cpu"

    Returns:
        Dict with:
            - "frames": All predicted frames including initial
            - "latents": Latent states at each step (if available)
    """

    all_frames = list(initial_frames)
    latents = []
    current_frames = initial_frames

    for action in actions:
        result = world_step(
            current_frames,
            action=action,
            model_path=model_path,
            model_type=model_type,
            num_steps=1,
            device=device,
            **kwargs
        )

        predicted = result["frames"]
        all_frames.extend(predicted)

        if result.get("latent") is not None:
            latents.append(result["latent"])

        # Slide window for next step
        current_frames = current_frames[len(predicted):] + predicted

    return {
        "frames": all_frames,
        "latents": latents if latents else None
    }


# ============== Model-specific implementations ==============

def _step_diamond(
    frames: List[np.ndarray],
    action: Optional[int],
    model_path: str,
    num_steps: int,
    device: str,
    num_actions: int = 18,  # Atari default
    denoising_steps: int = 3,  # DIAMOND default
    **kwargs
) -> Dict[str, Any]:
    """
    DIAMOND: Diffusion for World Modeling
    https://github.com/eloialonso/diamond

    Trained on Atari games, uses diffusion to predict next frames.

    Setup:
        git clone https://github.com/eloialonso/diamond.git
        cd diamond
        pip install -r requirements.txt

    Pretrained checkpoints downloaded via:
        python src/play.py --pretrained

    Args:
        frames: List of 4 recent frames as (H, W, C) uint8 arrays (64x64 for Atari)
        action: Discrete action index (0 to num_actions-1)
        model_path: Path to checkpoint .pt file
        num_steps: Number of frames to predict (rolls out autoregressively)
        device: "cuda" or "cpu"
        num_actions: Action space size (18 for Atari)
        denoising_steps: Diffusion denoising steps (3 is DIAMOND default)
    """
    import torch
    import sys
    from pathlib import Path

    # Add diamond src to path if needed
    diamond_path = kwargs.get("diamond_src_path")
    if diamond_path and diamond_path not in sys.path:
        sys.path.insert(0, diamond_path)

    try:
        from agent import Agent
        from envs.world_model_env import WorldModelEnv
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except ImportError as e:
        raise ImportError(
            f"DIAMOND dependencies not found: {e}. "
            "Clone https://github.com/eloialonso/diamond and add src/ to path, "
            "or pass diamond_src_path kwarg."
        )

    # Load agent from checkpoint
    ckpt_path = Path(model_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Load checkpoint to get config
    ckpt = torch.load(ckpt_path, map_location=device)

    # Build agent from config in checkpoint
    cfg = ckpt.get("config")
    if cfg is None:
        raise ValueError("Checkpoint missing config - ensure it's a DIAMOND checkpoint")

    agent_cfg = OmegaConf.create(cfg["agent"]) if isinstance(cfg, dict) else cfg.agent
    agent = Agent(instantiate(agent_cfg, num_actions=num_actions)).to(device).eval()
    agent.load(ckpt_path)

    # Prepare input frames
    # DIAMOND expects (B, T, C, H, W) with T=4 context frames, normalized to [-0.5, 0.5]
    if len(frames) < 4:
        # Pad with first frame if not enough history
        frames = [frames[0]] * (4 - len(frames)) + list(frames)
    frames = frames[-4:]  # Take last 4

    # Convert to tensor
    frames_t = []
    for f in frames:
        if f.dtype != np.uint8:
            f = (f * 255).astype(np.uint8)
        # Normalize to [-0.5, 0.5]
        f_norm = f.astype(np.float32) / 255.0 - 0.5
        # (H, W, C) -> (C, H, W)
        if f_norm.shape[-1] in [1, 3]:
            f_norm = np.transpose(f_norm, (2, 0, 1))
        frames_t.append(torch.from_numpy(f_norm))

    obs_buffer = torch.stack(frames_t, dim=0).unsqueeze(0).to(device)  # (1, 4, C, H, W)

    # Action as tensor
    if action is None:
        action = 0
    act_tensor = torch.tensor([[action]], dtype=torch.long, device=device)  # (1, 1)

    predicted_frames = []

    with torch.no_grad():
        for step in range(num_steps):
            # Sample next observation using diffusion
            # The denoiser expects: obs (B, T, C, H, W), act (B, T)
            next_obs = agent.denoiser.sample(
                obs=obs_buffer,
                act=act_tensor.expand(-1, obs_buffer.shape[1]),
                n_steps=denoising_steps
            )  # Returns (B, C, H, W)

            # Convert back to numpy
            pred_np = next_obs[0].cpu().numpy()  # (C, H, W)
            pred_np = np.transpose(pred_np, (1, 2, 0))  # (H, W, C)
            pred_np = ((pred_np + 0.5) * 255).clip(0, 255).astype(np.uint8)
            predicted_frames.append(pred_np)

            # Roll buffer for next step
            obs_buffer = torch.cat([
                obs_buffer[:, 1:],
                next_obs.unsqueeze(1)
            ], dim=1)

    return {
        "frames": predicted_frames,
        "latent": None,
        "metadata": {
            "model_type": "diamond",
            "denoising_steps": denoising_steps,
            "num_actions": num_actions
        }
    }


def _step_gamengen(
    frames: List[np.ndarray],
    action: Optional[int],
    model_path: str,
    num_steps: int,
    device: str,
    **kwargs
) -> Dict[str, Any]:
    """
    GameNGen: Neural game engine (e.g., DOOM in a neural net)
    https://gamengen.github.io/

    Uses diffusion conditioned on actions to generate game frames.
    """
    # TODO: Implement when open weights/code available
    #
    # GameNGen architecture:
    # - Takes ~64 previous frames for context
    # - Action conditioning via cross-attention
    # - Stable Diffusion backbone with game-specific fine-tuning

    raise NotImplementedError(
        "GameNGen support not yet implemented. "
        "No open weights currently available."
    )


def _step_dreamer(
    frames: List[np.ndarray],
    action: Optional[np.ndarray],
    model_path: str,
    num_steps: int,
    device: str,
    **kwargs
) -> Dict[str, Any]:
    """
    DreamerV3: Latent world model
    https://github.com/danijar/dreamerv3

    Operates in latent space, good for RL and planning.
    """
    # TODO: Implement with dreamerv3 package
    #
    # import dreamerv3
    #
    # agent = dreamerv3.Agent.load(model_path)
    #
    # # Encode current observation to latent
    # latent = agent.wm.encoder(frames[-1])
    #
    # # Predict forward in latent space
    # predicted_latents = []
    # current = latent
    # for _ in range(num_steps):
    #     current = agent.wm.dynamics.img_step(current, action)
    #     predicted_latents.append(current)
    #
    # # Decode back to images
    # predicted_frames = [agent.wm.decoder(l) for l in predicted_latents]
    #
    # return {"frames": predicted_frames, "latent": current, "metadata": {}}

    raise NotImplementedError(
        "DreamerV3 support not yet implemented. "
        "See https://github.com/danijar/dreamerv3 for setup."
    )


# ============== Setup/Download ==============

DIAMOND_REPO = "https://github.com/eloialonso/diamond.git"
DIAMOND_GAMES = ["Asterix", "Breakout", "Boxing", "Pong", "Seaquest", "SpaceInvaders"]


def setup_diamond(
    install_path: str = None,
    games: List[str] = None,
    device: str = "cuda"
) -> str:
    """
    Clone DIAMOND repo and download pretrained checkpoints.

    Args:
        install_path: Where to clone the repo. Defaults to ~/.npcpy/diamond
        games: List of games to download. Defaults to all available.
        device: "cuda" or "cpu" for checkpoint download

    Returns:
        Path to diamond/src directory (add to sys.path)
    """
    import subprocess
    import os

    if install_path is None:
        install_path = os.path.expanduser("~/.npcpy/diamond")

    diamond_dir = os.path.join(install_path, "diamond")
    src_path = os.path.join(diamond_dir, "src")

    # Clone if not exists
    if not os.path.exists(diamond_dir):
        print(f"Cloning DIAMOND to {diamond_dir}...")
        os.makedirs(install_path, exist_ok=True)
        subprocess.run(
            ["git", "clone", DIAMOND_REPO],
            cwd=install_path,
            check=True
        )
        print("Installing requirements...")
        subprocess.run(
            ["pip", "install", "-r", "requirements.txt"],
            cwd=diamond_dir,
            check=True
        )
    else:
        print(f"DIAMOND already exists at {diamond_dir}")

    # Download pretrained checkpoints
    checkpoints_dir = os.path.join(diamond_dir, "checkpoints")
    if not os.path.exists(checkpoints_dir) or not os.listdir(checkpoints_dir):
        print("Downloading pretrained checkpoints...")
        # DIAMOND's play.py --pretrained downloads them
        import sys
        sys.path.insert(0, src_path)

        # Download via their utility
        try:
            from utils import download_and_unzip_from_google_drive
            # Their checkpoint IDs (from their play.py)
            gdrive_ids = {
                "Asterix": "1KVvh0E2pFYGLdSU2dKiYntRXM3wPB7R7",
                "Boxing": "1_yy3ZcOa7mPCNBPyu1vGO-oiMpwzX8qT",
                "Breakout": "1VYKyFnEo0_pI9kAZmvJ5p0MdIp9u0gdj",
                "Pong": "1FBB-L8t-LzBV9wPZHCB9HUQRYP2K-9Xw",
                "Seaquest": "1nhEG9HHfQlCYVadG9TFbzQdHAqQ0G_Ty",
                "SpaceInvaders": "1N8rp7SYAalJkJT-2RQmfiU4UNZHUhm0W",
            }

            os.makedirs(checkpoints_dir, exist_ok=True)
            games_to_download = games or DIAMOND_GAMES

            for game in games_to_download:
                if game not in gdrive_ids:
                    print(f"Unknown game: {game}, skipping")
                    continue
                ckpt_path = os.path.join(checkpoints_dir, game)
                if os.path.exists(ckpt_path):
                    print(f"{game} checkpoint already exists")
                    continue
                print(f"Downloading {game}...")
                download_and_unzip_from_google_drive(gdrive_ids[game], checkpoints_dir)

        except Exception as e:
            print(f"Auto-download failed: {e}")
            print("Run manually: cd {diamond_dir} && python src/play.py --pretrained")

    return src_path


def get_diamond_checkpoint(game: str, install_path: str = None) -> str:
    """
    Get path to a DIAMOND checkpoint for a specific game.

    Args:
        game: One of Asterix, Boxing, Breakout, Pong, Seaquest, SpaceInvaders
        install_path: DIAMOND install location. Defaults to ~/.npcpy/diamond

    Returns:
        Path to checkpoint file
    """
    import os

    if install_path is None:
        install_path = os.path.expanduser("~/.npcpy/diamond")

    ckpt_dir = os.path.join(install_path, "diamond", "checkpoints", game)

    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(
            f"Checkpoint for {game} not found at {ckpt_dir}. "
            f"Run setup_diamond() first or download manually."
        )

    # Find the .pt file
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pt"):
            return os.path.join(ckpt_dir, f)

    # Sometimes it's in a subdirectory
    for root, dirs, files in os.walk(ckpt_dir):
        for f in files:
            if f.endswith(".pt"):
                return os.path.join(root, f)

    raise FileNotFoundError(f"No .pt checkpoint found in {ckpt_dir}")


def get_diamond_src_path(install_path: str = None) -> str:
    """Get path to diamond/src for imports."""
    import os
    if install_path is None:
        install_path = os.path.expanduser("~/.npcpy/diamond")
    return os.path.join(install_path, "diamond", "src")


def diamond_step(
    frames: List[np.ndarray],
    action: int,
    game: str = "Breakout",
    num_steps: int = 1,
    device: str = "cuda",
    auto_setup: bool = True
) -> Dict[str, Any]:
    """
    High-level API: predict next frame(s) using DIAMOND.

    Automatically handles setup, checkpoint loading, and caching.

    Args:
        frames: List of recent frames (at least 1, ideally 4). 64x64 uint8 RGB.
        action: Atari action index (0-17)
        game: One of Asterix, Boxing, Breakout, Pong, Seaquest, SpaceInvaders
        num_steps: Number of frames to predict
        device: "cuda" or "cpu"
        auto_setup: If True, automatically clone repo and download checkpoints

    Returns:
        Dict with "frames" (list of predicted np arrays) and "metadata"

    Example:
        >>> from npcpy.gen.world_gen import diamond_step
        >>> import numpy as np
        >>> frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        >>> result = diamond_step([frame], action=1, game="Breakout")
        >>> next_frame = result["frames"][0]
    """
    import os

    # Auto-setup if needed
    src_path = get_diamond_src_path()
    if auto_setup and not os.path.exists(src_path):
        print(f"DIAMOND not found, running setup...")
        src_path = setup_diamond(games=[game])

    # Get checkpoint
    try:
        ckpt_path = get_diamond_checkpoint(game)
    except FileNotFoundError:
        if auto_setup:
            print(f"Checkpoint for {game} not found, downloading...")
            setup_diamond(games=[game])
            ckpt_path = get_diamond_checkpoint(game)
        else:
            raise

    return world_step(
        frames=frames,
        action=action,
        model_path=ckpt_path,
        model_type="diamond",
        num_steps=num_steps,
        device=device,
        diamond_src_path=src_path
    )


# ============== Utilities ==============

# Global cache for loaded models
_MODEL_CACHE: Dict[str, Any] = {}


def load_diamond_model(
    model_path: str,
    device: str = "cuda",
    num_actions: int = 18,
    diamond_src_path: str = None
):
    """
    Pre-load a DIAMOND model for faster repeated inference.

    Args:
        model_path: Path to checkpoint .pt file
        device: "cuda" or "cpu"
        num_actions: Action space size
        diamond_src_path: Path to diamond/src directory

    Returns:
        Tuple of (agent, denoising_config)
    """
    import torch
    import sys
    from pathlib import Path

    cache_key = f"diamond:{model_path}:{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if diamond_src_path and diamond_src_path not in sys.path:
        sys.path.insert(0, diamond_src_path)

    from agent import Agent
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    ckpt_path = Path(model_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config")

    agent_cfg = OmegaConf.create(cfg["agent"]) if isinstance(cfg, dict) else cfg.agent
    agent = Agent(instantiate(agent_cfg, num_actions=num_actions)).to(device).eval()
    agent.load(ckpt_path)

    _MODEL_CACHE[cache_key] = agent
    return agent


def clear_model_cache():
    """Clear all cached models to free memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()


def load_world_model(model_path: str, model_type: str, device: str = "cuda", **kwargs):
    """
    Pre-load a world model for faster repeated inference.

    Args:
        model_path: Path to checkpoint
        model_type: "diamond", "gamengen", or "dreamer"
        device: "cuda" or "cpu"

    Returns:
        Loaded model object (type depends on model_type)
    """
    if model_type == "diamond":
        return load_diamond_model(model_path, device, **kwargs)
    else:
        raise NotImplementedError(f"Model loading not implemented for {model_type}")


def frames_to_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30
) -> str:
    """Save predicted frames to video file."""
    import cv2
    import os

    if not frames:
        raise ValueError("No frames to save")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()
    return output_path
