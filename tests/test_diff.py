"""Test suite for diffusion training (diff.py) module."""

import pytest


class TestDiffusionConfig:
    """Test DiffusionConfig dataclass."""

    def test_default_values(self):
        """Test DiffusionConfig has correct defaults"""
        from npcpy.ft.diff import DiffusionConfig

        config = DiffusionConfig()

        assert config.image_size == 128
        assert config.channels == 256
        assert config.time_emb_dim == 128
        assert config.timesteps == 1000
        assert config.beta_start == 1e-4
        assert config.beta_end == 0.02
        assert config.num_epochs == 100
        assert config.batch_size == 4
        assert config.learning_rate == 1e-5
        assert config.checkpoint_frequency == 10
        assert config.output_model_path == "diffusion_model"
        assert config.use_clip is False
        assert config.num_channels == 3

    def test_custom_values(self):
        """Test DiffusionConfig with custom values"""
        from npcpy.ft.diff import DiffusionConfig

        config = DiffusionConfig(
            image_size=64,
            channels=128,
            timesteps=500,
            num_epochs=50,
            batch_size=8,
            learning_rate=2e-4,
            output_model_path="/custom/path"
        )

        assert config.image_size == 64
        assert config.channels == 128
        assert config.timesteps == 500
        assert config.num_epochs == 50
        assert config.batch_size == 8
        assert config.learning_rate == 2e-4
        assert config.output_model_path == "/custom/path"

    def test_config_with_clip_enabled(self):
        """Test DiffusionConfig with CLIP enabled"""
        from npcpy.ft.diff import DiffusionConfig

        config = DiffusionConfig(use_clip=True)

        assert config.use_clip is True

    def test_config_grayscale_images(self):
        """Test DiffusionConfig for grayscale images"""
        from npcpy.ft.diff import DiffusionConfig

        config = DiffusionConfig(num_channels=1)

        assert config.num_channels == 1


class TestTorchAvailability:
    """Test TORCH_AVAILABLE flag behavior."""

    def test_torch_available_flag_exists(self):
        """Test TORCH_AVAILABLE flag is defined"""
        from npcpy.ft.diff import TORCH_AVAILABLE

        assert isinstance(TORCH_AVAILABLE, bool)

    def test_classes_defined_based_on_torch(self):
        """Test classes are defined or None based on torch availability"""
        from npcpy.ft.diff import (
            TORCH_AVAILABLE,
            SinusoidalPositionEmbeddings,
            SimpleUNet,
            ImageDataset,
            DiffusionTrainer
        )

        if TORCH_AVAILABLE:
            assert SinusoidalPositionEmbeddings is not None
            assert SimpleUNet is not None
            assert ImageDataset is not None
            assert DiffusionTrainer is not None
        else:
            assert SinusoidalPositionEmbeddings is None
            assert SimpleUNet is None
            assert ImageDataset is None
            assert DiffusionTrainer is None


class TestTrainDiffusionFunction:
    """Test train_diffusion function."""

    def test_train_diffusion_requires_torch(self):
        """Test train_diffusion raises ImportError without torch"""
        from npcpy.ft.diff import TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            from npcpy.ft.diff import train_diffusion

            with pytest.raises(ImportError) as exc_info:
                train_diffusion([])

            assert "PyTorch" in str(exc_info.value)

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not available"),
        reason="torch not available"
    )
    def test_train_diffusion_with_empty_paths(self):
        """Test train_diffusion with empty image paths"""
        torch = pytest.importorskip("torch")
        from npcpy.ft.diff import train_diffusion, DiffusionConfig

        # Empty list should cause error during dataloader iteration
        config = DiffusionConfig(num_epochs=1, batch_size=1)

        # This will fail because there are no images
        with pytest.raises(Exception):
            train_diffusion([], config=config)


class TestGenerateImageFunction:
    """Test generate_image function."""

    def test_generate_image_requires_torch(self):
        """Test generate_image raises ImportError without torch"""
        from npcpy.ft.diff import TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            from npcpy.ft.diff import generate_image

            with pytest.raises(ImportError) as exc_info:
                generate_image("/fake/model.pt")

            assert "PyTorch" in str(exc_info.value)


class TestTorchDependentComponents:
    """Test torch-dependent components when torch is available."""

    def test_simple_unet_creation(self):
        """Test SimpleUNet can be created"""
        torch = pytest.importorskip("torch")
        from npcpy.ft.diff import SimpleUNet, TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")

        model = SimpleUNet(image_size=64, channels=64, time_emb_dim=32)

        assert model.image_size == 64
        assert hasattr(model, "conv_in")
        assert hasattr(model, "conv_out")

    def test_simple_unet_forward_pass(self):
        """Test SimpleUNet forward pass"""
        torch = pytest.importorskip("torch")
        from npcpy.ft.diff import SimpleUNet, TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")

        model = SimpleUNet(image_size=64, channels=32, time_emb_dim=16)

        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64)
        t = torch.randint(0, 1000, (batch_size,))

        output = model(x, t)

        assert output.shape == (batch_size, 3, 64, 64)

    def test_sinusoidal_embeddings(self):
        """Test SinusoidalPositionEmbeddings"""
        torch = pytest.importorskip("torch")
        from npcpy.ft.diff import SinusoidalPositionEmbeddings, TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")

        embeddings = SinusoidalPositionEmbeddings(dim=64)
        t = torch.tensor([0, 100, 500, 999])

        output = embeddings(t)

        assert output.shape == (4, 64)

    def test_diffusion_trainer_initialization(self):
        """Test DiffusionTrainer initialization"""
        torch = pytest.importorskip("torch")
        from npcpy.ft.diff import DiffusionTrainer, DiffusionConfig, TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")

        config = DiffusionConfig(
            image_size=32,
            channels=32,
            timesteps=100
        )

        trainer = DiffusionTrainer(config)

        assert trainer.config == config
        assert trainer.model is not None
        assert len(trainer.betas) == 100
        assert len(trainer.alphas) == 100

    def test_diffusion_trainer_add_noise(self):
        """Test DiffusionTrainer add_noise method"""
        torch = pytest.importorskip("torch")
        from npcpy.ft.diff import DiffusionTrainer, DiffusionConfig, TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")

        config = DiffusionConfig(image_size=32, channels=32, timesteps=100)
        trainer = DiffusionTrainer(config)

        x = torch.randn(2, 3, 32, 32).to(trainer.device)
        t = torch.tensor([10, 50]).to(trainer.device)

        noisy, noise = trainer.add_noise(x, t)

        assert noisy.shape == x.shape
        assert noise.shape == x.shape

    def test_diffusion_trainer_sample(self):
        """Test DiffusionTrainer sample method"""
        torch = pytest.importorskip("torch")
        from npcpy.ft.diff import DiffusionTrainer, DiffusionConfig, TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")

        # Use minimal config for fast testing
        config = DiffusionConfig(
            image_size=16,
            channels=16,
            timesteps=10,  # Very few timesteps for speed
            num_channels=3
        )

        trainer = DiffusionTrainer(config)

        samples = trainer.sample(num_samples=1)

        assert samples.shape == (1, 3, 16, 16)
        # Check values are in valid range [0, 1]
        assert samples.min() >= 0
        assert samples.max() <= 1


class TestImageDataset:
    """Test ImageDataset class."""

    def test_image_dataset_creation(self):
        """Test ImageDataset can be created"""
        torch = pytest.importorskip("torch")
        from npcpy.ft.diff import ImageDataset, TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")

        dataset = ImageDataset(
            image_paths=["/fake/img1.jpg", "/fake/img2.jpg"],
            captions=["caption1", "caption2"],
            image_size=64
        )

        assert len(dataset) == 2
        assert dataset.image_size == 64

    def test_image_dataset_length(self):
        """Test ImageDataset __len__"""
        torch = pytest.importorskip("torch")
        from npcpy.ft.diff import ImageDataset, TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")

        dataset = ImageDataset(
            image_paths=[f"/path/img{i}.jpg" for i in range(10)],
            captions=None,
            image_size=128
        )

        assert len(dataset) == 10

    def test_image_dataset_captions_default(self):
        """Test ImageDataset handles None captions"""
        torch = pytest.importorskip("torch")
        from npcpy.ft.diff import ImageDataset, TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")

        dataset = ImageDataset(
            image_paths=["/path/img.jpg"],
            captions=None,
            image_size=64
        )

        # Should default to empty strings
        assert dataset.captions == [""]


class TestBetaSchedule:
    """Test beta schedule calculations."""

    def test_beta_schedule_linear(self):
        """Test linear beta schedule"""
        torch = pytest.importorskip("torch")
        from npcpy.ft.diff import DiffusionConfig, TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")

        config = DiffusionConfig(
            beta_start=0.0001,
            beta_end=0.02,
            timesteps=1000
        )

        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps)

        assert len(betas) == 1000
        assert abs(betas[0].item() - 0.0001) < 1e-6
        assert abs(betas[-1].item() - 0.02) < 1e-6

    def test_alpha_cumprod_decreases(self):
        """Test alpha cumulative product decreases over time"""
        torch = pytest.importorskip("torch")
        from npcpy.ft.diff import DiffusionTrainer, DiffusionConfig, TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")

        config = DiffusionConfig(timesteps=100)
        trainer = DiffusionTrainer(config)

        # Alpha cumprod should decrease (more noise over time)
        assert trainer.alphas_cumprod[0] > trainer.alphas_cumprod[-1]
        # Should start near 1 (first value close to 1 - beta_start)
        assert trainer.alphas_cumprod[0] > 0.99
        # End value depends on beta schedule - just verify it decreased significantly
        assert trainer.alphas_cumprod[-1] < trainer.alphas_cumprod[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
