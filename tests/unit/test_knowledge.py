"""Unit tests for knowledge distillation engine."""

import pytest
import torch
import torch.nn as nn

from multi_agent_collab_learning.core.knowledge import KnowledgeDistillationEngine


class TestKnowledgeDistillationEngine:
    """Test KnowledgeDistillationEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a knowledge distillation engine for testing."""
        return KnowledgeDistillationEngine(
            input_dim=64,
            compressed_dim=32,
            num_agents=4
        )

    @pytest.fixture
    def sample_knowledge(self):
        """Create sample knowledge tensor."""
        return torch.randn(1, 64)

    @pytest.mark.unit
    def test_engine_initialization(self, engine):
        """Test proper engine initialization."""
        assert engine.input_dim == 64
        assert engine.compressed_dim == 32
        assert engine.num_agents == 4

        # Check network architectures
        assert isinstance(engine.compression_network, nn.Sequential)
        assert isinstance(engine.decompression_network, nn.Sequential)
        assert isinstance(engine.quality_assessor, nn.Sequential)
        assert isinstance(engine.cross_agent_alignment, nn.Sequential)

    @pytest.mark.unit
    def test_compression_network_structure(self, engine):
        """Test compression network has correct structure."""
        compression_layers = list(engine.compression_network.children())

        # Should have: Linear(64, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 32)
        assert len(compression_layers) == 5
        assert isinstance(compression_layers[0], nn.Linear)
        assert compression_layers[0].in_features == 64
        assert compression_layers[0].out_features == 128
        assert isinstance(compression_layers[4], nn.Linear)
        assert compression_layers[4].out_features == 32

    @pytest.mark.unit
    def test_decompression_network_structure(self, engine):
        """Test decompression network has correct structure."""
        decompression_layers = list(engine.decompression_network.children())

        # Should have: Linear(32, 64) -> ReLU -> Linear(64, 128) -> ReLU -> Linear(128, 64)
        assert len(decompression_layers) == 5
        assert isinstance(decompression_layers[0], nn.Linear)
        assert decompression_layers[0].in_features == 32
        assert decompression_layers[0].out_features == 64
        assert isinstance(decompression_layers[4], nn.Linear)
        assert decompression_layers[4].out_features == 64

    @pytest.mark.unit
    def test_compress_knowledge(self, engine, sample_knowledge):
        """Test knowledge compression."""
        compressed = engine.compress_knowledge(sample_knowledge)

        assert compressed.shape == (1, 32)
        assert compressed.dtype == torch.float32
        assert not torch.isnan(compressed).any()
        assert not torch.isinf(compressed).any()

    @pytest.mark.unit
    def test_decompress_knowledge(self, engine, sample_knowledge):
        """Test knowledge decompression."""
        compressed = engine.compress_knowledge(sample_knowledge)
        decompressed = engine.decompress_knowledge(compressed)

        assert decompressed.shape == sample_knowledge.shape
        assert decompressed.dtype == torch.float32
        assert not torch.isnan(decompressed).any()
        assert not torch.isinf(decompressed).any()

    @pytest.mark.unit
    def test_compression_decompression_cycle(self, engine, sample_knowledge):
        """Test full compression-decompression cycle."""
        original = sample_knowledge.clone()
        compressed = engine.compress_knowledge(original)
        reconstructed = engine.decompress_knowledge(compressed)

        # Should maintain shape
        assert reconstructed.shape == original.shape

        # Reconstruction should be reasonably close (allowing for compression loss)
        mse_loss = torch.nn.functional.mse_loss(reconstructed, original)
        assert mse_loss < 5.0  # Allow for some reconstruction error

    @pytest.mark.unit
    def test_assess_quality(self, engine, sample_knowledge):
        """Test knowledge quality assessment."""
        quality_score = engine.assess_quality(sample_knowledge)

        assert quality_score.shape == (1, 1)
        assert 0 <= quality_score.item() <= 1  # Should be a probability
        assert not torch.isnan(quality_score).any()

    @pytest.mark.unit
    def test_align_cross_agent(self, engine):
        """Test cross-agent alignment."""
        # Create sample agent representations
        agent_representations = torch.randn(4, 32)  # 4 agents, 32-dim compressed knowledge

        alignment_matrix = engine.align_cross_agent(agent_representations)

        assert alignment_matrix.shape == (4, 4)  # 4x4 alignment matrix
        assert not torch.isnan(alignment_matrix).any()
        assert not torch.isinf(alignment_matrix).any()

    @pytest.mark.unit
    def test_transfer_knowledge(self, engine, sample_knowledge):
        """Test complete knowledge transfer process."""
        source_agent_id = 0
        target_agent_id = 1

        transfer_result = engine.transfer_knowledge(
            source_knowledge=sample_knowledge,
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id
        )

        # Check result structure
        assert 'transferred_knowledge' in transfer_result
        assert 'transfer_quality' in transfer_result
        assert 'compression_ratio' in transfer_result
        assert 'alignment_score' in transfer_result

        # Check transferred knowledge
        transferred = transfer_result['transferred_knowledge']
        assert transferred.shape == sample_knowledge.shape
        assert not torch.isnan(transferred).any()

        # Check quality metrics
        assert 0 <= transfer_result['transfer_quality'] <= 1
        assert transfer_result['compression_ratio'] > 0
        assert not torch.isnan(torch.tensor(transfer_result['alignment_score'])).any()

    @pytest.mark.unit
    def test_batch_knowledge_processing(self, engine):
        """Test processing batch of knowledge samples."""
        batch_size = 8
        batch_knowledge = torch.randn(batch_size, 64)

        # Test batch compression
        compressed_batch = engine.compress_knowledge(batch_knowledge)
        assert compressed_batch.shape == (batch_size, 32)

        # Test batch decompression
        decompressed_batch = engine.decompress_knowledge(compressed_batch)
        assert decompressed_batch.shape == (batch_size, 64)

        # Test batch quality assessment
        quality_scores = engine.assess_quality(batch_knowledge)
        assert quality_scores.shape == (batch_size, 1)

    @pytest.mark.unit
    def test_gradient_flow(self, engine, sample_knowledge):
        """Test that gradients flow properly through the networks."""
        # Enable gradient computation
        sample_knowledge.requires_grad_(True)

        # Forward pass
        compressed = engine.compress_knowledge(sample_knowledge)
        decompressed = engine.decompress_knowledge(compressed)
        quality = engine.assess_quality(decompressed)

        # Compute loss and backward pass
        loss = torch.nn.functional.mse_loss(decompressed, sample_knowledge) + (1 - quality).mean()
        loss.backward()

        # Check that gradients exist
        assert sample_knowledge.grad is not None
        assert not torch.isnan(sample_knowledge.grad).any()

        # Check that network parameters have gradients
        for param in engine.compression_network.parameters():
            if param.requires_grad:
                assert param.grad is not None

    @pytest.mark.unit
    def test_different_input_dimensions(self):
        """Test engine with different input dimensions."""
        # Test with smaller dimensions
        small_engine = KnowledgeDistillationEngine(
            input_dim=32,
            compressed_dim=16,
            num_agents=2
        )

        small_input = torch.randn(1, 32)
        compressed = small_engine.compress_knowledge(small_input)
        assert compressed.shape == (1, 16)

        # Test with larger dimensions
        large_engine = KnowledgeDistillationEngine(
            input_dim=128,
            compressed_dim=64,
            num_agents=8
        )

        large_input = torch.randn(1, 128)
        compressed = large_engine.compress_knowledge(large_input)
        assert compressed.shape == (1, 64)

    @pytest.mark.unit
    def test_invalid_inputs(self, engine):
        """Test handling of invalid inputs."""
        # Test wrong input dimension
        wrong_dim_input = torch.randn(1, 32)  # Should be 64
        with pytest.raises(RuntimeError):
            engine.compress_knowledge(wrong_dim_input)

        # Test wrong compressed dimension for decompression
        wrong_compressed = torch.randn(1, 16)  # Should be 32
        with pytest.raises(RuntimeError):
            engine.decompress_knowledge(wrong_compressed)

    @pytest.mark.unit
    def test_deterministic_behavior(self, deterministic_environment):
        """Test that engine produces deterministic results with fixed seeds."""
        engine1 = KnowledgeDistillationEngine(input_dim=64, compressed_dim=32, num_agents=4)
        engine2 = KnowledgeDistillationEngine(input_dim=64, compressed_dim=32, num_agents=4)

        sample_input = torch.randn(1, 64)

        # Both engines should produce the same output with same weights initialization
        torch.manual_seed(42)
        result1 = engine1.compress_knowledge(sample_input)

        torch.manual_seed(42)
        result2 = engine2.compress_knowledge(sample_input)

        # Results should be identical with same initialization
        assert torch.allclose(result1, result2, atol=1e-6)

    @pytest.mark.unit
    def test_knowledge_transfer_metrics(self, engine, sample_knowledge):
        """Test that transfer metrics are reasonable."""
        transfer_result = engine.transfer_knowledge(
            source_knowledge=sample_knowledge,
            source_agent_id=0,
            target_agent_id=1
        )

        # Compression ratio should be reasonable (input_dim / compressed_dim)
        expected_compression_ratio = 64 / 32
        assert abs(transfer_result['compression_ratio'] - expected_compression_ratio) < 0.1

        # Transfer quality should be between 0 and 1
        assert 0 <= transfer_result['transfer_quality'] <= 1

        # Alignment score should be finite
        assert not torch.isnan(torch.tensor(transfer_result['alignment_score'])).any()
        assert not torch.isinf(torch.tensor(transfer_result['alignment_score'])).any()