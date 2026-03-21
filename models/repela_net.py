"""
RepELA-Net: Reparameterized Efficient Linear Attention Network
for Lightweight 2D Material Segmentation.

Complete model assembly combining:
  1. Stem (initial convolution + color space enhancement)
  2. RepConv Stages (stages 1-2, local features)
  3. ELA Stages (stages 3-4, global features)
  4. DW-MFF Decoder (dynamic weighted multi-scale fusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rep_conv import RepConvStage
from .ela_block import ELAStage
from .decoder import DWMFFDecoder


class ColorSpaceEnhancement(nn.Module):
    """Color Space Enhancement Module.
    
    MoS2 material images have subtle color differences between layers.
    This module extracts the Saturation channel from HSV color space
    and concatenates it with RGB, providing more discriminative color features.
    
    Input: RGB [B, 3, H, W]
    Output: Enhanced [B, 4, H, W] (RGB + S_channel)
    """

    def __init__(self):
        super().__init__()
        # Learnable weight for the saturation channel
        self.s_weight = nn.Parameter(torch.tensor(1.0))

    def rgb_to_saturation(self, rgb):
        """Extract saturation channel from RGB image.
        
        Saturation = (Max - Min) / Max, where Max and Min are
        per-pixel max/min across R, G, B channels.
        """
        max_rgb = rgb.max(dim=1, keepdim=True)[0]
        min_rgb = rgb.min(dim=1, keepdim=True)[0]
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
        return saturation

    def forward(self, x):
        # x: [B, 3, H, W] normalized RGB
        sat = self.rgb_to_saturation(x)  # [B, 1, H, W]
        # Concatenate RGB with weighted saturation
        enhanced = torch.cat([x, self.s_weight * sat], dim=1)
        return enhanced


class RepELANet(nn.Module):
    """RepELA-Net: Complete lightweight segmentation model.
    
    Architecture overview:
        Input (512x512x3)
        -> ColorSpaceEnhancement (512x512x4)
        -> Stem Conv (256x256xC1=32)
        -> RepConv Stage 1 (128x128xC1=32)
        -> RepConv Stage 2 (64x64xC2=64)
        -> ELA Stage 3 (32x32xC3=128)
        -> ELA Stage 4 (16x16xC4=256)
        -> DW-MFF Decoder (128x128x4)
        -> Upsample (512x512x4)
    
    Target: <3M params, <2G FLOPs @512x512
    """

    def __init__(self, num_classes=4, channels=(32, 64, 128, 256),
                 num_blocks=(2, 2, 4, 2), num_heads=(0, 0, 4, 8),
                 decoder_channels=128, drop_rate=0.1, deploy=False,
                 deep_supervision=False):
        """
        Args:
            num_classes: number of segmentation classes (4 for MoS2 dataset)
            channels: channel count at each stage
            num_blocks: number of blocks at each stage
            num_heads: attention heads at each stage (0 = RepConv, >0 = ELA)
            decoder_channels: channel count in decoder
            drop_rate: dropout rate
            deploy: if True, use fused/deployed convolutions
            deep_supervision: if True, add auxiliary classification heads
        """
        super().__init__()
        self.num_classes = num_classes
        c1, c2, c3, c4 = channels

        # Color space enhancement
        self.color_enhance = ColorSpaceEnhancement()

        # Stem: initial downsampling (4x4 -> 1/4 resolution)
        self.stem = nn.Sequential(
            nn.Conv2d(4, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )

        # Stage 1: RepConv (1/4 -> 1/8)
        self.stage1 = RepConvStage(
            c1, c1, num_blocks=num_blocks[0],
            expand_ratio=2, use_se=True, deploy=deploy
        )

        # Stage 2: RepConv (1/8 -> 1/16)
        self.stage2 = RepConvStage(
            c1, c2, num_blocks=num_blocks[1],
            expand_ratio=2, use_se=True, deploy=deploy
        )

        # Stage 3: ELA (1/16 -> 1/32)
        self.stage3 = ELAStage(
            c2, c3, num_blocks=num_blocks[2],
            num_heads=num_heads[2], expand_ratio=2,
            drop=drop_rate, attn_drop=0., scales=(1, 2, 4)
        )

        # Stage 4: ELA (1/32 -> 1/64)
        self.stage4 = ELAStage(
            c3, c4, num_blocks=num_blocks[3],
            num_heads=num_heads[3], expand_ratio=2,
            drop=drop_rate, attn_drop=0., scales=(1, 2)
        )

        # Decoder
        self.decoder = DWMFFDecoder(
            in_channels_list=[c1, c2, c3, c4],
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            deep_supervision=deep_supervision
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: input image [B, 3, H, W]
        Returns:
            logits: [B, num_classes, H, W]
        """
        input_size = x.shape[2:]

        # Color space enhancement
        x = self.color_enhance(x)  # [B, 4, H, W]

        # Stem
        x = self.stem(x)  # [B, C1, H/2, W/2]

        # Encoder stages
        f1 = self.stage1(x)   # [B, C1, H/4, W/4]
        f2 = self.stage2(f1)  # [B, C2, H/8, W/8]
        f3 = self.stage3(f2)  # [B, C3, H/16, W/16]
        f4 = self.stage4(f3)  # [B, C4, H/32, W/32]

        # Decoder
        decoder_out = self.decoder([f1, f2, f3, f4])

        if isinstance(decoder_out, tuple):
            # Deep supervision: (main_logits, [aux3, aux2])
            out, aux_list = decoder_out
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
            aux_list = [F.interpolate(a, size=input_size, mode='bilinear', align_corners=False)
                        for a in aux_list]
            return out, aux_list
        else:
            out = decoder_out
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
            return out

    def switch_to_deploy(self):
        """Switch all reparameterizable modules to deploy/inference mode."""
        self.stage1.switch_to_deploy()
        self.stage2.switch_to_deploy()
        print("[RepELA-Net] Switched to deploy mode (reparameterized)")

    def get_params_flops(self, input_size=(512, 512)):
        """Compute parameter count and FLOPs."""
        from thop import profile, clever_format
        dummy = torch.randn(1, 3, *input_size).to(next(self.parameters()).device)
        flops, params = profile(self, inputs=(dummy,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.2f")
        return params, flops, params_str, flops_str


def repela_net_tiny(num_classes=4, deploy=False, deep_supervision=False):
    """RepELA-Net-Tiny: ~1.5M params."""
    return RepELANet(
        num_classes=num_classes,
        channels=(24, 48, 96, 192),
        num_blocks=(2, 2, 3, 2),
        num_heads=(0, 0, 4, 8),
        decoder_channels=96,
        drop_rate=0.05,
        deploy=deploy,
        deep_supervision=deep_supervision
    )


def repela_net_small(num_classes=4, deploy=False, deep_supervision=False):
    """RepELA-Net-Small: ~2.8M params (recommended for MoS2)."""
    return RepELANet(
        num_classes=num_classes,
        channels=(32, 64, 128, 256),
        num_blocks=(2, 2, 4, 2),
        num_heads=(0, 0, 4, 8),
        decoder_channels=128,
        drop_rate=0.1,
        deploy=deploy,
        deep_supervision=deep_supervision
    )


def repela_net_base(num_classes=4, deploy=False, deep_supervision=False):
    """RepELA-Net-Base: ~5M params (higher accuracy)."""
    return RepELANet(
        num_classes=num_classes,
        channels=(48, 96, 192, 384),
        num_blocks=(2, 2, 6, 2),
        num_heads=(0, 0, 6, 12),
        decoder_channels=192,
        drop_rate=0.1,
        deploy=deploy,
        deep_supervision=deep_supervision
    )
