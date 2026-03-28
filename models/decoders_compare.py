"""
Comparison Decoder Implementations for RepELA-Net Encoder.

All decoders take the same input interface:
    features: [f1, f2, f3, f4]
        f1: [B, C1, H/4,  W/4]   (C1=32)
        f2: [B, C2, H/8,  W/8]   (C2=64)
        f3: [B, C3, H/16, W/16]  (C3=128)
        f4: [B, C4, H/32, W/32]  (C4=256)

    Returns: logits [B, num_classes, H/4, W/4]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# Helper blocks
# ═══════════════════════════════════════════════════════════════════════

class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════════════
# 1. UNet Decoder
# ═══════════════════════════════════════════════════════════════════════

class UNetDecoder(nn.Module):
    """Classic UNet skip-connection decoder with progressive upsampling."""

    def __init__(self, in_channels_list, num_classes=4, decoder_ch=128):
        super().__init__()
        c1, c2, c3, c4 = in_channels_list  # 32, 64, 128, 256

        # Upsampling blocks: deep → shallow
        self.up4 = nn.Sequential(
            ConvBNReLU(c4, decoder_ch),
            ConvBNReLU(decoder_ch, decoder_ch),
        )
        self.up3 = nn.Sequential(
            ConvBNReLU(decoder_ch + c3, decoder_ch),
            ConvBNReLU(decoder_ch, decoder_ch),
        )
        self.up2 = nn.Sequential(
            ConvBNReLU(decoder_ch + c2, decoder_ch // 2),
            ConvBNReLU(decoder_ch // 2, decoder_ch // 2),
        )
        self.up1 = nn.Sequential(
            ConvBNReLU(decoder_ch // 2 + c1, decoder_ch // 4),
            ConvBNReLU(decoder_ch // 4, decoder_ch // 4),
        )
        self.seg_head = nn.Conv2d(decoder_ch // 4, num_classes, 1)

    def forward(self, features):
        f1, f2, f3, f4 = features

        x = self.up4(f4)
        x = F.interpolate(x, size=f3.shape[2:], mode='bilinear', align_corners=False)
        x = self.up3(torch.cat([x, f3], dim=1))

        x = F.interpolate(x, size=f2.shape[2:], mode='bilinear', align_corners=False)
        x = self.up2(torch.cat([x, f2], dim=1))

        x = F.interpolate(x, size=f1.shape[2:], mode='bilinear', align_corners=False)
        x = self.up1(torch.cat([x, f1], dim=1))

        return self.seg_head(x)


# ═══════════════════════════════════════════════════════════════════════
# 2. FPN Decoder
# ═══════════════════════════════════════════════════════════════════════

class FPNDecoder(nn.Module):
    """Feature Pyramid Network decoder with lateral connections."""

    def __init__(self, in_channels_list, num_classes=4, fpn_ch=128):
        super().__init__()
        c1, c2, c3, c4 = in_channels_list

        # Lateral convolutions (1x1 to unify channels)
        self.lat4 = nn.Conv2d(c4, fpn_ch, 1)
        self.lat3 = nn.Conv2d(c3, fpn_ch, 1)
        self.lat2 = nn.Conv2d(c2, fpn_ch, 1)
        self.lat1 = nn.Conv2d(c1, fpn_ch, 1)

        # Smooth convolutions (3x3 after addition)
        self.smooth4 = ConvBNReLU(fpn_ch, fpn_ch)
        self.smooth3 = ConvBNReLU(fpn_ch, fpn_ch)
        self.smooth2 = ConvBNReLU(fpn_ch, fpn_ch)
        self.smooth1 = ConvBNReLU(fpn_ch, fpn_ch)

        # Merge all levels
        self.merge = ConvBNReLU(fpn_ch * 4, fpn_ch)
        self.seg_head = nn.Conv2d(fpn_ch, num_classes, 1)

    def forward(self, features):
        f1, f2, f3, f4 = features

        # Top-down pathway
        p4 = self.lat4(f4)
        p3 = self.lat3(f3) + F.interpolate(p4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        p1 = self.lat1(f1) + F.interpolate(p2, size=f1.shape[2:], mode='bilinear', align_corners=False)

        # Smooth
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        # Upsample all to f1 resolution and merge
        target = f1.shape[2:]
        merged = torch.cat([
            p1,
            F.interpolate(p2, size=target, mode='bilinear', align_corners=False),
            F.interpolate(p3, size=target, mode='bilinear', align_corners=False),
            F.interpolate(p4, size=target, mode='bilinear', align_corners=False),
        ], dim=1)

        return self.seg_head(self.merge(merged))


# ═══════════════════════════════════════════════════════════════════════
# 3. DeepLabV3+ ASPP Decoder
# ═══════════════════════════════════════════════════════════════════════

class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling."""

    def __init__(self, c_in, c_out, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            ConvBNReLU(c_in, c_out, k=1, p=0),  # 1x1 conv
        ])
        for rate in rates:
            self.branches.append(
                ConvBNReLU(c_in, c_out, k=3, p=rate, groups=1)
                if rate == 0 else
                nn.Sequential(
                    nn.Conv2d(c_in, c_out, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.GELU(),
                )
            )
        # Global average pooling branch (use GroupNorm to avoid BN issue with 1x1 spatial)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in, c_out, 1, bias=False),
            nn.GroupNorm(min(32, c_out), c_out),
            nn.GELU(),
        )
        self.project = ConvBNReLU(c_out * (len(rates) + 2), c_out, k=1, p=0)

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        gap = self.gap(x)
        gap = F.interpolate(gap, size=x.shape[2:], mode='bilinear', align_corners=False)
        outs.append(gap)
        return self.project(torch.cat(outs, dim=1))


class ASPPDecoder(nn.Module):
    """DeepLabV3+ style: ASPP on deepest feature + low-level skip."""

    def __init__(self, in_channels_list, num_classes=4, aspp_ch=128):
        super().__init__()
        c1, c2, c3, c4 = in_channels_list

        self.aspp = ASPPModule(c4, aspp_ch)
        self.low_level_conv = ConvBNReLU(c1, 48, k=1, p=0)
        self.fuse = nn.Sequential(
            ConvBNReLU(aspp_ch + 48, aspp_ch),
            ConvBNReLU(aspp_ch, aspp_ch),
        )
        self.seg_head = nn.Conv2d(aspp_ch, num_classes, 1)

    def forward(self, features):
        f1, f2, f3, f4 = features

        # ASPP on deepest features
        x = self.aspp(f4)
        x = F.interpolate(x, size=f1.shape[2:], mode='bilinear', align_corners=False)

        # Low-level skip connection from f1
        low = self.low_level_conv(f1)
        x = torch.cat([x, low], dim=1)
        x = self.fuse(x)

        return self.seg_head(x)


# ═══════════════════════════════════════════════════════════════════════
# 4. PSPNet PPM Decoder
# ═══════════════════════════════════════════════════════════════════════

class PPMDecoder(nn.Module):
    """Pyramid Pooling Module (PSPNet) decoder."""

    def __init__(self, in_channels_list, num_classes=4, ppm_ch=128, bins=(1, 2, 3, 6)):
        super().__init__()
        c1, c2, c3, c4 = in_channels_list

        self.ppm = nn.ModuleList()
        for b in bins:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                ConvBNReLU(c4, ppm_ch // len(bins), k=1, p=0),
            ))
        ppm_out = c4 + (ppm_ch // len(bins)) * len(bins)

        self.bottleneck = ConvBNReLU(ppm_out, ppm_ch)
        self.seg_head = nn.Conv2d(ppm_ch, num_classes, 1)

    def forward(self, features):
        f1, f2, f3, f4 = features
        h, w = f4.shape[2:]

        ppm_outs = [f4]
        for pool in self.ppm:
            p = pool(f4)
            p = F.interpolate(p, size=(h, w), mode='bilinear', align_corners=False)
            ppm_outs.append(p)

        x = torch.cat(ppm_outs, dim=1)
        x = self.bottleneck(x)

        # Upsample to f1 resolution
        x = F.interpolate(x, size=f1.shape[2:], mode='bilinear', align_corners=False)
        return self.seg_head(x)


# ═══════════════════════════════════════════════════════════════════════
# 5. SegFormer MLP Head
# ═══════════════════════════════════════════════════════════════════════

class SegFormerMLPDecoder(nn.Module):
    """SegFormer-style All-MLP decoder: align channels → upsample → concat → fuse."""

    def __init__(self, in_channels_list, num_classes=4, embed_dim=128):
        super().__init__()
        self.align = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, embed_dim, 1), nn.BatchNorm2d(embed_dim))
            for c in in_channels_list
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.seg_head = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features):
        f1, f2, f3, f4 = features
        target = f1.shape[2:]

        aligned = []
        for i, (feat, align) in enumerate(zip(features, self.align)):
            x = align(feat)
            if x.shape[2:] != target:
                x = F.interpolate(x, size=target, mode='bilinear', align_corners=False)
            aligned.append(x)

        x = self.fuse(torch.cat(aligned, dim=1))
        return self.seg_head(x)


# ═══════════════════════════════════════════════════════════════════════
# 6. Hamburger Decoder (Matrix Decomposition)
# ═══════════════════════════════════════════════════════════════════════

class NMF(nn.Module):
    """Non-negative Matrix Factorization module (Hamburger core)."""

    def __init__(self, channels, rank=64, num_iters=6):
        super().__init__()
        self.rank = rank
        self.num_iters = num_iters
        self.channels = channels

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape to [B, C, N] where N = H*W
        x_flat = x.view(B, C, -1)  # [B, C, N]
        x_flat = F.relu(x_flat)

        # Initialize bases and coefficients
        if self.training:
            bases = torch.rand(B, C, self.rank, device=x.device)
            coefs = torch.rand(B, self.rank, H * W, device=x.device)
        else:
            bases = torch.ones(B, C, self.rank, device=x.device)
            coefs = torch.ones(B, self.rank, H * W, device=x.device)

        # Multiplicative update rules
        for _ in range(self.num_iters):
            # Update coefs: coefs *= (bases^T @ x) / (bases^T @ bases @ coefs + eps)
            numerator = torch.bmm(bases.transpose(1, 2), x_flat)
            denominator = torch.bmm(torch.bmm(bases.transpose(1, 2), bases), coefs) + 1e-6
            coefs = coefs * numerator / denominator

            # Update bases: bases *= (x @ coefs^T) / (bases @ coefs @ coefs^T + eps)
            numerator = torch.bmm(x_flat, coefs.transpose(1, 2))
            denominator = torch.bmm(bases, torch.bmm(coefs, coefs.transpose(1, 2))) + 1e-6
            bases = bases * numerator / denominator

        # Reconstruct
        x_recon = torch.bmm(bases, coefs)  # [B, C, N]
        return x_recon.view(B, C, H, W)


class HamburgerDecoder(nn.Module):
    """Hamburger decoder: align → concat → NMF matrix decomposition → classify."""

    def __init__(self, in_channels_list, num_classes=4, ham_ch=128):
        super().__init__()
        self.align = nn.ModuleList([
            ConvBNReLU(c, ham_ch, k=1, p=0) for c in in_channels_list
        ])
        self.pre_ham = ConvBNReLU(ham_ch * 4, ham_ch, k=1, p=0)
        self.ham = NMF(ham_ch, rank=64, num_iters=6)
        self.post_ham = ConvBNReLU(ham_ch, ham_ch)
        self.seg_head = nn.Conv2d(ham_ch, num_classes, 1)

    def forward(self, features):
        f1, f2, f3, f4 = features
        target = f1.shape[2:]

        aligned = []
        for feat, align in zip(features, self.align):
            x = align(feat)
            if x.shape[2:] != target:
                x = F.interpolate(x, size=target, mode='bilinear', align_corners=False)
            aligned.append(x)

        x = self.pre_ham(torch.cat(aligned, dim=1))
        x_ham = self.ham(x)
        x = x + x_ham  # Residual
        x = self.post_ham(x)
        return self.seg_head(x)


# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════

DECODER_REGISTRY = {
    'unet':      UNetDecoder,
    'fpn':       FPNDecoder,
    'aspp':      ASPPDecoder,
    'ppm':       PPMDecoder,
    'segformer': SegFormerMLPDecoder,
    'hamburger': HamburgerDecoder,
}

DECODER_NAMES = {
    'unet':      'UNet Decoder',
    'fpn':       'FPN Decoder',
    'aspp':      'DeepLabV3+ ASPP Decoder',
    'ppm':       'PSPNet PPM Decoder',
    'segformer': 'SegFormer MLP Head',
    'hamburger': 'Hamburger Decoder',
}


def build_decoder(name, in_channels_list, num_classes=4):
    """Build a decoder by name."""
    if name not in DECODER_REGISTRY:
        raise ValueError(f'Unknown decoder: {name}. Available: {list(DECODER_REGISTRY.keys())}')
    return DECODER_REGISTRY[name](in_channels_list, num_classes=num_classes)
