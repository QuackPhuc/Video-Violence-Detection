import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConvBlock3D(nn.Module):
    """3D convolutional block with spatial and temporal convolutions.

    Applies spatial convolution followed by temporal convolution,
    each with batch normalization and activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_kernel: Tuple[int, int, int] = (1, 3, 3),
        temporal_kernel: Tuple[int, int, int] = (3, 1, 1),
        spatial_padding: Tuple[int, int, int] = (0, 1, 1),
        temporal_padding: Tuple[int, int, int] = (1, 0, 0),
        bias: bool = False,
    ):
        super().__init__()

        # Spatial convolution
        self.spatial_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=spatial_kernel,
            padding=spatial_padding,
            bias=bias,
        )
        self.spatial_bn = nn.BatchNorm3d(out_channels)

        # Temporal convolution
        self.temporal_conv = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=temporal_kernel,
            padding=temporal_padding,
            bias=bias,
        )
        self.temporal_bn = nn.BatchNorm3d(out_channels)

        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, activation: str = "relu") -> torch.Tensor:
        """Forward pass through the convolutional block.

        Args:
            x: Input tensor of shape (B, C, T, H, W)
            activation: Activation function to use ('relu' or 'sigmoid')

        Returns:
            Output tensor of shape (B, C', T, H, W)
        """
        # Spatial pathway
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)

        if activation == "relu":
            x = self.relu(x)
        elif activation == "sigmoid":
            x = self.sigmoid(x)

        # Temporal pathway
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)

        if activation == "relu":
            x = self.relu(x)
        elif activation == "sigmoid":
            x = self.sigmoid(x)

        return x


class StreamBlock(nn.Module):
    """Stream block containing multiple ConvBlock3D layers.

    Processes features through three consecutive convolutional blocks
    with configurable channel dimensions.
    """

    def __init__(self, channels: Tuple[int, int, int]):
        """Initialize stream block.

        Args:
            channels: Tuple of (input_channels, mid_channels, output_channels)
        """
        super().__init__()
        in_ch, mid_ch, out_ch = channels

        self.conv1 = ConvBlock3D(in_ch, mid_ch)
        self.conv2 = ConvBlock3D(mid_ch, mid_ch)
        self.conv3 = ConvBlock3D(mid_ch, out_ch)

    def forward(self, x: torch.Tensor, final_activation: str = "relu") -> torch.Tensor:
        """Forward pass through the stream block.

        Args:
            x: Input tensor
            final_activation: Activation for the final layer

        Returns:
            Processed tensor
        """
        x = self.conv1(x, activation="relu")
        x = self.conv2(x, activation="relu")
        x = self.conv3(x, activation=final_activation)
        return x


class TwoStreamGated3DCNN(nn.Module):
    """Two-stream 3D CNN for video violence detection.

    Architecture:
    - RGB stream: Processes appearance information
    - Optical flow stream: Processes motion information
    - Gated fusion: Multiplicative fusion of streams
    - Classification head: Final classification layers

    Input: 6-channel tensor (3 RGB + 2 flow + 1 magnitude)
    Output: Classification logits
    """

    def __init__(
        self,
        num_classes: int = 1,
        dropout_prob: float = 0.5,
        feature_channels: Tuple[int, ...] = (32, 64, 128, 256),
        freeze_backbones: bool = False,
    ):
        """Initialize the two-stream model.

        Args:
            num_classes: Number of output classes
            dropout_prob: Dropout probability
            feature_channels: Channel dimensions for each level
            freeze_backbones: Whether to freeze backbone parameters (unused)
        """
        super().__init__()

        self.num_classes = num_classes
        self.feature_channels = feature_channels

        # RGB Stream
        self.rgb_stream = nn.ModuleDict(
            {
                "block1": StreamBlock((3, 16, feature_channels[0])),
                "block2": StreamBlock(
                    (feature_channels[0], feature_channels[0], feature_channels[1])
                ),
                "block3": StreamBlock(
                    (feature_channels[1], feature_channels[1], feature_channels[2])
                ),
            }
        )

        # Optical Flow Stream
        self.flow_stream = nn.ModuleDict(
            {
                "block1": StreamBlock((3, 16, feature_channels[0])),
                "block2": StreamBlock(
                    (feature_channels[0], feature_channels[0], feature_channels[1])
                ),
                "block3": StreamBlock(
                    (feature_channels[1], feature_channels[1], feature_channels[2])
                ),
            }
        )

        # Pooling layers
        self.pool = nn.ModuleDict(
            {
                "pool1": nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                "pool2": nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                "pool3": nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                "fusion_pool": nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                "merged_pool": nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            }
        )

        # Merged stream blocks
        self.merged_stream = nn.ModuleDict(
            {
                "block1": StreamBlock(
                    (feature_channels[2], feature_channels[2], feature_channels[3])
                ),
                "block2": StreamBlock(
                    (feature_channels[3], feature_channels[3], feature_channels[3])
                ),
                "block3": StreamBlock(
                    (feature_channels[3], feature_channels[3], feature_channels[3])
                ),
            }
        )

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(feature_channels[3], feature_channels[2] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(feature_channels[2] // 2, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

        logger.info(
            f"Initialized TwoStreamGated3DCNN with {self._count_parameters():,} parameters"
        )

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (B, 6, T, H, W)
               Channels 0-2: RGB, Channels 3-5: Flow

        Returns:
            Classification logits of shape (B, num_classes)
        """
        # Split input into RGB and flow streams
        rgb = x[:, :3, :, :, :]
        flow = x[:, 3:, :, :, :]

        # RGB stream processing
        rgb = self.rgb_stream["block1"](rgb)
        rgb = self.pool["pool1"](rgb)

        rgb = self.rgb_stream["block2"](rgb)
        rgb = self.pool["pool2"](rgb)

        rgb = self.rgb_stream["block3"](rgb)
        rgb = self.pool["pool3"](rgb)

        # Optical flow stream processing
        flow = self.flow_stream["block1"](flow)
        flow = self.pool["pool1"](flow)

        flow = self.flow_stream["block2"](flow)
        flow = self.pool["pool2"](flow)

        # Final flow block uses sigmoid for gating
        flow = self.flow_stream["block3"](flow, final_activation="sigmoid")
        flow = self.pool["pool3"](flow)

        # Gated fusion
        fused = rgb * flow
        fused = self.pool["fusion_pool"](fused)

        # Merged stream processing
        fused = self.merged_stream["block1"](fused)
        fused = self.pool["merged_pool"](fused)

        fused = self.merged_stream["block2"](fused)
        fused = self.pool["merged_pool"](fused)

        fused = self.merged_stream["block3"](fused)

        # Global pooling and classification
        fused = self.global_pool(fused)
        fused = fused.view(fused.size(0), -1)

        logits = self.classifier(fused)

        return logits

    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract intermediate feature maps for visualization.

        Args:
            x: Input tensor

        Returns:
            Dictionary of feature maps from different layers
        """
        feature_maps = {}

        # Split input
        rgb = x[:, :3, :, :, :]
        flow = x[:, 3:, :, :, :]

        # RGB stream
        rgb = self.rgb_stream["block1"](rgb)
        feature_maps["rgb_block1"] = rgb.detach()
        rgb = self.pool["pool1"](rgb)

        rgb = self.rgb_stream["block2"](rgb)
        feature_maps["rgb_block2"] = rgb.detach()
        rgb = self.pool["pool2"](rgb)

        rgb = self.rgb_stream["block3"](rgb)
        feature_maps["rgb_block3"] = rgb.detach()
        rgb = self.pool["pool3"](rgb)

        # Flow stream
        flow = self.flow_stream["block1"](flow)
        feature_maps["flow_block1"] = flow.detach()
        flow = self.pool["pool1"](flow)

        flow = self.flow_stream["block2"](flow)
        feature_maps["flow_block2"] = flow.detach()
        flow = self.pool["pool2"](flow)

        flow = self.flow_stream["block3"](flow, final_activation="sigmoid")
        feature_maps["flow_block3"] = flow.detach()
        flow = self.pool["pool3"](flow)

        # Fusion
        fused = rgb * flow
        feature_maps["fused"] = fused.detach()

        return feature_maps
