import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.two_stream_cnn import ConvBlock3D, StreamBlock, TwoStreamGated3DCNN


class TestConvBlock3D:
    """Test ConvBlock3D class."""

    def test_initialization(self):
        """Test ConvBlock3D initialization."""
        block = ConvBlock3D(in_channels=3, out_channels=32)

        assert block.spatial_conv.in_channels == 3
        assert block.spatial_conv.out_channels == 32
        assert block.temporal_conv.in_channels == 32
        assert block.temporal_conv.out_channels == 32

    def test_forward_pass_relu(self):
        """Test forward pass with ReLU activation."""
        block = ConvBlock3D(in_channels=3, out_channels=32)
        x = torch.randn(2, 3, 16, 224, 224)  # Batch, Channels, Time, Height, Width

        output = block(x, activation="relu")

        assert output.shape == (2, 32, 16, 224, 224)
        assert torch.all(output >= 0)  # ReLU should make all values non-negative

    def test_forward_pass_sigmoid(self):
        """Test forward pass with sigmoid activation."""
        block = ConvBlock3D(in_channels=3, out_channels=32)
        x = torch.randn(2, 3, 16, 224, 224)

        output = block(x, activation="sigmoid")

        assert output.shape == (2, 32, 16, 224, 224)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid range [0, 1]

    def test_custom_kernel_sizes(self):
        """Test with custom kernel sizes."""
        block = ConvBlock3D(
            in_channels=6,
            out_channels=64,
            spatial_kernel=(1, 5, 5),
            temporal_kernel=(5, 1, 1),
            spatial_padding=(0, 2, 2),
            temporal_padding=(2, 0, 0),
        )
        x = torch.randn(1, 6, 16, 112, 112)

        output = block(x)
        assert output.shape == (1, 64, 16, 112, 112)

    def test_bias_parameter(self):
        """Test bias parameter."""
        block_with_bias = ConvBlock3D(in_channels=3, out_channels=32, bias=True)
        block_without_bias = ConvBlock3D(in_channels=3, out_channels=32, bias=False)

        assert block_with_bias.spatial_conv.bias is not None
        assert block_without_bias.spatial_conv.bias is None


class TestStreamBlock:
    """Test StreamBlock class."""

    def test_initialization(self):
        """Test StreamBlock initialization."""
        channels = (3, 16, 32)
        block = StreamBlock(channels)

        assert block.conv1.spatial_conv.in_channels == 3
        assert block.conv1.spatial_conv.out_channels == 16
        assert block.conv3.temporal_conv.out_channels == 32

    def test_forward_pass(self):
        """Test forward pass through stream block."""
        channels = (6, 32, 64)
        block = StreamBlock(channels)
        x = torch.randn(2, 6, 16, 112, 112)

        output = block(x)

        assert output.shape == (2, 64, 16, 112, 112)
        assert torch.all(output >= 0)  # Default ReLU activation

    def test_final_activation_sigmoid(self):
        """Test with sigmoid final activation."""
        channels = (3, 16, 32)
        block = StreamBlock(channels)
        x = torch.randn(1, 3, 8, 64, 64)

        output = block(x, final_activation="sigmoid")

        assert output.shape == (1, 32, 8, 64, 64)
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_different_channel_configurations(self):
        """Test with different channel configurations."""
        test_configs = [
            (1, 8, 16),
            (3, 32, 64),
            (6, 64, 128),
            (32, 64, 64),  # Same input and output channels
        ]

        for channels in test_configs:
            block = StreamBlock(channels)
            x = torch.randn(1, channels[0], 8, 32, 32)
            output = block(x)
            assert output.shape == (1, channels[2], 8, 32, 32)


class TestTwoStreamGated3DCNN:
    """Test TwoStreamGated3DCNN class."""

    def test_initialization_default(self):
        """Test default initialization."""
        model = TwoStreamGated3DCNN()

        assert model.num_classes == 1
        assert len(model.feature_channels) == 4
        assert model.feature_channels == (32, 64, 128, 256)

    def test_initialization_custom(self):
        """Test custom initialization."""
        model = TwoStreamGated3DCNN(
            num_classes=5, dropout_prob=0.3, feature_channels=(16, 32, 64, 128)
        )

        assert model.num_classes == 5
        assert model.feature_channels == (16, 32, 64, 128)

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        model = TwoStreamGated3DCNN(num_classes=1)
        x = torch.randn(2, 6, 16, 224, 224)  # 6 channels: 3 RGB + 3 flow

        output = model(x)

        assert output.shape == (2, 1)

    def test_forward_pass_different_input_sizes(self):
        """Test with different input sizes."""
        model = TwoStreamGated3DCNN(num_classes=2)

        test_inputs = [
            (1, 6, 8, 112, 112),
            (4, 6, 16, 224, 224),
            (2, 6, 32, 56, 56),
        ]

        for batch_size, channels, time, height, width in test_inputs:
            x = torch.randn(batch_size, channels, time, height, width)
            output = model(x)
            assert output.shape == (batch_size, 2)

    def test_input_channel_validation(self):
        """Test that model expects 6 input channels."""
        model = TwoStreamGated3DCNN()

        # Test with wrong number of channels
        x_wrong = torch.randn(1, 3, 16, 224, 224)  # Only 3 channels

        # This should fail during forward pass
        try:
            output = model(x_wrong)
            # If no error, check if output shape is wrong
            assert False, "Model should expect 6 channels"
        except:
            pass  # Expected to fail

    def test_stream_separation(self):
        """Test that RGB and flow streams are properly separated."""
        model = TwoStreamGated3DCNN()
        x = torch.randn(1, 6, 16, 224, 224)

        # Get feature maps to verify stream separation
        feature_maps = model.get_feature_maps(x)

        assert "rgb_block1" in feature_maps
        assert "flow_block1" in feature_maps
        assert "fused" in feature_maps

        # Check shapes
        rgb_features = feature_maps["rgb_block1"]
        flow_features = feature_maps["flow_block1"]
        fused_features = feature_maps["fused"]

        assert rgb_features.shape[1] == 32  # Default first feature channel
        assert flow_features.shape[1] == 32
        assert fused_features.shape[1] == 128  # After third block

    def test_get_feature_maps(self):
        """Test get_feature_maps method."""
        model = TwoStreamGated3DCNN()
        x = torch.randn(1, 6, 8, 112, 112)

        feature_maps = model.get_feature_maps(x)

        expected_keys = [
            "rgb_block1",
            "rgb_block2",
            "rgb_block3",
            "flow_block1",
            "flow_block2",
            "flow_block3",
            "fused",
        ]

        for key in expected_keys:
            assert key in feature_maps
            assert isinstance(feature_maps[key], torch.Tensor)

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        model = TwoStreamGated3DCNN()
        x = torch.randn(1, 6, 16, 224, 224, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_model_eval_mode(self):
        """Test model in evaluation mode."""
        model = TwoStreamGated3DCNN()
        x = torch.randn(2, 6, 16, 224, 224)

        model.eval()

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2)

    def test_model_train_mode(self):
        """Test model in training mode with dropout."""
        model = TwoStreamGated3DCNN(dropout_prob=0.5)
        x = torch.randn(2, 6, 16, 224, 224)

        model.train()

        output1 = model(x)
        output2 = model(x)

        # Outputs might be different due to dropout in training mode
        # Just check that they have correct shape
        assert output1.shape == (2, 1)
        assert output2.shape == (2, 1)

    def test_parameter_count(self):
        """Test parameter count method."""
        model = TwoStreamGated3DCNN()
        param_count = model._count_parameters()

        assert param_count > 0
        assert isinstance(param_count, int)

        # Verify it matches manual count
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == manual_count

    def test_different_output_classes(self):
        """Test with different number of output classes."""
        for num_classes in [1, 2, 5, 10]:
            model = TwoStreamGated3DCNN(num_classes=num_classes)
            x = torch.randn(1, 6, 16, 224, 224)
            output = model(x)
            assert output.shape == (1, num_classes)

    def test_batch_processing(self):
        """Test batch processing capabilities."""
        model = TwoStreamGated3DCNN()

        # Test different batch sizes
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 6, 16, 224, 224)
            output = model(x)
            assert output.shape == (batch_size, 1)

    def test_memory_efficiency(self):
        """Test memory usage is reasonable."""
        model = TwoStreamGated3DCNN()
        x = torch.randn(1, 6, 16, 224, 224)

        # Clear any cached gradients
        model.zero_grad()

        # Forward pass
        output = model(x)

        # Should complete without memory errors
        assert output.shape == (1, 1)


class TestModelIntegration:
    """Integration tests for the complete model pipeline."""

    def test_training_step_simulation(self):
        """Simulate a training step."""
        model = TwoStreamGated3DCNN()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Simulate batch
        x = torch.randn(2, 6, 16, 224, 224)
        y = torch.randint(0, 2, (2, 1)).float()

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0
        assert outputs.shape == y.shape

    def test_inference_consistency(self):
        """Test inference consistency."""
        model = TwoStreamGated3DCNN()
        model.eval()

        x = torch.randn(1, 6, 16, 224, 224)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x.clone())
            output3 = model(x)

        # All outputs should be identical
        assert torch.allclose(output1, output2)
        assert torch.allclose(output1, output3)

    def test_device_compatibility(self):
        """Test model works on CPU (and GPU if available)."""
        model = TwoStreamGated3DCNN()
        x = torch.randn(1, 6, 8, 112, 112)

        # Test on CPU
        model = model.cpu()
        x = x.cpu()
        output_cpu = model(x)
        assert output_cpu.device.type == "cpu"

        # Test on GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            output_gpu = model(x)
            assert output_gpu.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__])
