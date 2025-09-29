from npcpy.ft.diffusion import (
    train_diffusion,
    generate_image,
    DiffusionConfig
)
from PIL import Image
import numpy as np
import os


def create_synthetic_data(output_dir="synthetic_images", num_images=100):
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = []
    captions = []
    
    for i in range(num_images):
        img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        
        if i % 3 == 0:
            img[30:100, 30:100] = 255
            caption = "white square on black background"
        elif i % 3 == 1:
            img[40:90, 40:90] = 255
            caption = "white circle shape"
        else:
            img[20:50, 20:110] = 255
            caption = "white rectangle horizontal"
        
        img_path = os.path.join(output_dir, f"img_{i}.png")
        Image.fromarray(img).save(img_path)
        
        image_paths.append(img_path)
        captions.append(caption)
    
    return image_paths, captions


def demo_train_diffusion():
    print("\n=== TRAIN DIFFUSION MODEL ===")
    
    print("Creating synthetic training data...")
    image_paths, captions = create_synthetic_data(num_images=50)
    
    config = DiffusionConfig(
        image_size=128,
        channels=128,
        num_epochs=5,
        batch_size=2,
        learning_rate=1e-4,
        checkpoint_frequency=100,
        output_dir="models/diffusion_shapes"
    )
    
    print("Training diffusion model...")
    model_path = train_diffusion(
        image_paths,
        captions,
        config
    )
    
    print(f"Model saved to: {model_path}")
    
    return model_path


def demo_generate_from_diffusion():
    print("\n=== GENERATE FROM DIFFUSION MODEL ===")
    
    model_path = "models/diffusion_shapes/final_model.pt"
    
    if not os.path.exists(model_path):
        print("No trained model found. Train first with demo_train_diffusion()")
        return
    
    prompt = "white square on black background"
    
    print(f"Generating image for: {prompt}")
    image = generate_image(
        model_path,
        prompt,
        image_size=128
    )
    
    image_np = image[0, 0].cpu().numpy()
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    
    output_path = "generated_image.png"
    image_pil.save(output_path)
    print(f"Saved to: {output_path}")
    
    return image_pil


def demo_resume_training():
    print("\n=== RESUME DIFFUSION TRAINING ===")
    
    image_paths, captions = create_synthetic_data(num_images=50)
    
    checkpoint_path = "models/diffusion_shapes/checkpoints/checkpoint-epoch0-step100.pt"
    
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Train from scratch first.")
        return
    
    config = DiffusionConfig(
        output_dir="models/diffusion_shapes",
        num_epochs=10
    )
    
    print(f"Resuming from: {checkpoint_path}")
    model_path = train_diffusion(
        image_paths,
        captions,
        config,
        resume_from=checkpoint_path
    )
    
    return model_path


if __name__ == "__main__":
    print("Diffusion Model Demo")
    print("=" * 50)
    
    demo_train_diffusion()
    
    demo_generate_from_diffusion()
    
    print("\n" + "=" * 50)
    print("Demo complete!")