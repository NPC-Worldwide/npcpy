"""Tests for embeddings fine-tuning module."""

import pytest
import numpy as np


class TestEmbeddingConfig:
    """Test EmbeddingConfig and HilbertConfig dataclasses."""

    def test_embedding_config_defaults(self):
        from npcpy.ft.embeddings import EmbeddingConfig

        config = EmbeddingConfig()
        assert config.base_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.output_model_path == "models/embedding"
        assert config.device == "cpu"
        assert config.embedding_dim == 384
        assert config.num_train_epochs == 10
        assert config.batch_size == 16
        assert config.learning_rate == 2e-5
        assert config.temperature == 0.07
        assert config.margin == 0.5
        assert config.loss_type == "infonce"
        assert config.max_length == 256

    def test_embedding_config_custom(self):
        from npcpy.ft.embeddings import EmbeddingConfig

        config = EmbeddingConfig(
            base_model_name="bert-base-uncased",
            output_model_path="custom/path",
            device="cuda",
            embedding_dim=768,
            loss_type="triplet",
        )
        assert config.base_model_name == "bert-base-uncased"
        assert config.output_model_path == "custom/path"
        assert config.device == "cuda"
        assert config.embedding_dim == 768
        assert config.loss_type == "triplet"

    def test_hilbert_config_defaults(self):
        from npcpy.ft.embeddings import HilbertConfig

        config = HilbertConfig()
        assert config.base_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.output_model_path == "models/hilbert_embedding"
        assert config.use_phase is True
        assert config.lambda_phase == 0.5
        assert config.phase_init_scale == 0.1
        assert config.loss_type == "hilbert_infonce"

    def test_hilbert_config_custom(self):
        from npcpy.ft.embeddings import HilbertConfig

        config = HilbertConfig(
            use_phase=False,
            lambda_phase=0.3,
            loss_type="phase_triplet",
        )
        assert config.use_phase is False
        assert config.lambda_phase == 0.3
        assert config.loss_type == "phase_triplet"


class TestHelpers:
    """Test internal helper functions."""

    def test_complex_tensor_init(self):
        from npcpy.ft.embeddings import ComplexTensor
        import torch

        mag = torch.randn(4, 10)
        ang = torch.randn(4, 10)
        ct = ComplexTensor(mag, ang)
        assert ct.magnitude is mag
        assert ct.angle is ang

    def test_hilbert_similarity_range(self):
        from npcpy.ft.embeddings import ComplexTensor, _hilbert_similarity
        import torch

        # Two identical states should have similarity 1
        mag = torch.ones(1, 5)
        ang = torch.zeros(1, 5)
        ct1 = ComplexTensor(mag, ang)
        ct2 = ComplexTensor(mag, ang)
        sim = _hilbert_similarity(ct1, ct2)
        assert sim.item() == pytest.approx(1.0, abs=1e-5)

        # Two orthogonal states (in phase terms) should have similarity 0
        ang2 = torch.ones(1, 5) * (np.pi / 2)
        ct3 = ComplexTensor(mag, ang2)
        sim2 = _hilbert_similarity(ct1, ct3)
        # Cos(pi/2) = 0, so real part should be near 0
        assert abs(sim2.item()) < 0.1

    def test_hilbert_similarity_matrix_shape(self):
        from npcpy.ft.embeddings import ComplexTensor, _hilbert_similarity_matrix
        import torch

        mag1 = torch.randn(3, 8)
        ang1 = torch.randn(3, 8)
        mag2 = torch.randn(5, 8)
        ang2 = torch.randn(5, 8)
        ct1 = ComplexTensor(mag1, ang1)
        ct2 = ComplexTensor(mag2, ang2)
        sim = _hilbert_similarity_matrix(ct1, ct2)
        assert sim.shape == (3, 5)

    def test_normalize_hilbert(self):
        from npcpy.ft.embeddings import ComplexTensor, _normalize_hilbert
        import torch

        mag = torch.randn(2, 6)
        ang = torch.randn(2, 6)
        ct = ComplexTensor(mag, ang)
        normed = _normalize_hilbert(ct)
        # Check norms are ~1
        norms = (normed.magnitude ** 2).sum(dim=-1).sqrt()
        assert norms[0].item() == pytest.approx(1.0, abs=1e-5)
        assert norms[1].item() == pytest.approx(1.0, abs=1e-5)


class TestTorchAvailability:
    """Test availability flags."""

    def test_torch_available_exists(self):
        from npcpy.ft.embeddings import TORCH_AVAILABLE
        assert isinstance(TORCH_AVAILABLE, bool)

    def test_mlx_available_exists(self):
        from npcpy.ft.embeddings import MLX_AVAILABLE
        assert isinstance(MLX_AVAILABLE, bool)


@pytest.mark.slow
class TestClassicalEmbeddingTraining:
    """Integration tests for classical embedding fine-tuning."""

    @pytest.fixture
    def dummy_data(self):
        anchors = ["This is sentence A", "This is sentence B", "This is sentence C"]
        positives = ["Sentence A paraphrase", "Sentence B paraphrase", "Sentence C paraphrase"]
        negatives = ["Unrelated sentence X", "Unrelated sentence Y", "Unrelated sentence Z"]
        return anchors, positives, negatives

    def test_run_embedding_sft_torch(self, tmp_path, dummy_data):
        from npcpy.ft.embeddings import run_embedding_sft_torch, EmbeddingConfig

        anchors, positives, negatives = dummy_data
        config = EmbeddingConfig(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            output_model_path=str(tmp_path / "test_embedding"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            loss_type="triplet",
            max_length=32,
        )
        path = run_embedding_sft_torch(anchors, positives, negatives, config=config)
        assert path == config.output_model_path
        import os
        assert os.path.exists(os.path.join(path, "model.pt"))

    def test_run_embedding_sft_infonce(self, tmp_path, dummy_data):
        from npcpy.ft.embeddings import run_embedding_sft_torch, EmbeddingConfig

        anchors, positives, negatives = dummy_data
        config = EmbeddingConfig(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            output_model_path=str(tmp_path / "test_embedding_infonce"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            loss_type="infonce",
            max_length=32,
        )
        path = run_embedding_sft_torch(anchors, positives, config=config)
        assert path == config.output_model_path

    def test_load_and_encode(self, tmp_path, dummy_data):
        from npcpy.ft.embeddings import (
            run_embedding_sft_torch, load_embedding_model, encode_texts, EmbeddingConfig
        )

        anchors, positives, negatives = dummy_data
        config = EmbeddingConfig(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            output_model_path=str(tmp_path / "test_embedding_load"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            loss_type="infonce",
            max_length=32,
        )
        run_embedding_sft_torch(anchors, positives, config=config)
        base, projector, tokenizer, loaded_config = load_embedding_model(
            config.output_model_path, device="cpu"
        )
        assert loaded_config.base_model_name == config.base_model_name
        assert loaded_config.embedding_dim == config.embedding_dim

        embs = encode_texts(["test sentence"], base, projector, tokenizer, device="cpu", max_length=32)
        assert len(embs) == 1
        assert len(embs[0]) == config.embedding_dim

    def test_evaluate_embeddings(self, tmp_path, dummy_data):
        from npcpy.ft.embeddings import (
            run_embedding_sft_torch, load_embedding_model, evaluate_embeddings, EmbeddingConfig
        )

        anchors, positives, negatives = dummy_data
        config = EmbeddingConfig(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            output_model_path=str(tmp_path / "test_embedding_eval"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            loss_type="triplet",
            max_length=32,
        )
        run_embedding_sft_torch(anchors, positives, negatives, config=config)
        base, projector, tokenizer, _ = load_embedding_model(config.output_model_path, device="cpu")
        metrics = evaluate_embeddings(
            anchors, positives, negatives,
            base, projector, tokenizer, device="cpu", max_length=32
        )
        assert "mrr" in metrics
        assert "recall@1" in metrics
        assert "recall@5" in metrics
        assert 0.0 <= metrics["mrr"] <= 1.0


@pytest.mark.slow
class TestHilbertEmbeddingTraining:
    """Integration tests for Hilbert-space embedding fine-tuning."""

    @pytest.fixture
    def dummy_data(self):
        anchors = ["This is sentence A", "This is sentence B", "This is sentence C"]
        positives = ["Sentence A paraphrase", "Sentence B paraphrase", "Sentence C paraphrase"]
        negatives = ["Unrelated sentence X", "Unrelated sentence Y", "Unrelated sentence Z"]
        return anchors, positives, negatives

    def test_run_hilbert_embedding_sft_torch(self, tmp_path, dummy_data):
        from npcpy.ft.embeddings import run_hilbert_embedding_sft_torch, HilbertConfig

        anchors, positives, negatives = dummy_data
        config = HilbertConfig(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            output_model_path=str(tmp_path / "test_hilbert"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            loss_type="hilbert_infonce",
            max_length=32,
        )
        path = run_hilbert_embedding_sft_torch(anchors, positives, config=config)
        assert path == config.output_model_path
        import os
        assert os.path.exists(os.path.join(path, "model.pt"))

    def test_run_hilbert_phase_triplet(self, tmp_path, dummy_data):
        from npcpy.ft.embeddings import run_hilbert_embedding_sft_torch, HilbertConfig

        anchors, positives, negatives = dummy_data
        config = HilbertConfig(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            output_model_path=str(tmp_path / "test_hilbert_triplet"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            loss_type="phase_triplet",
            max_length=32,
        )
        path = run_hilbert_embedding_sft_torch(anchors, positives, negatives, config=config)
        assert path == config.output_model_path


class TestFoundationModelTraining:
    """Tests for training embeddings from scratch (no pretrained weights)."""

    @pytest.fixture
    def dummy_data(self):
        anchors = ["This is sentence A", "This is sentence B", "This is sentence C"]
        positives = ["Sentence A paraphrase", "Sentence B paraphrase", "Sentence C paraphrase"]
        negatives = ["Unrelated sentence X", "Unrelated sentence Y", "Unrelated sentence Z"]
        return anchors, positives, negatives

    @pytest.mark.slow
    def test_train_from_scratch_classical(self, tmp_path, dummy_data):
        from npcpy.ft.embeddings import run_embedding_sft_torch, EmbeddingConfig

        anchors, positives, negatives = dummy_data
        config = EmbeddingConfig(
            base_model_name="prajjwal1/bert-tiny",  # Tiny model for fast testing
            output_model_path=str(tmp_path / "test_scratch"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            loss_type="infonce",
            max_length=32,
            embedding_dim=128,
        )
        path = run_embedding_sft_torch(anchors, positives, config=config)
        assert path == config.output_model_path

    @pytest.mark.slow
    def test_train_from_scratch_hilbert(self, tmp_path, dummy_data):
        from npcpy.ft.embeddings import run_hilbert_embedding_sft_torch, HilbertConfig

        anchors, positives, negatives = dummy_data
        config = HilbertConfig(
            base_model_name="prajjwal1/bert-tiny",
            output_model_path=str(tmp_path / "test_hilbert_scratch"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            loss_type="hilbert_infonce",
            max_length=32,
            embedding_dim=128,
        )
        path = run_hilbert_embedding_sft_torch(anchors, positives, config=config)
        assert path == config.output_model_path
