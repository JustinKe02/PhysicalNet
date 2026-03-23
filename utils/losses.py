"""
Loss Functions for RepELA-Net.

Hybrid loss: Focal Loss + Dice Loss + optional Boundary Loss
Designed to handle severe class imbalance in MoS2 dataset.

Class distribution:
  background: 74.86%
  multilayer: 19.56%
  monolayer:   3.12%
  fewlayer:    2.46%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Numerically stable implementation that works under AMP (float16).
    """

    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W]
            targets: [B, H, W] with class indices
        """
        num_classes = logits.shape[1]

        # Cast to float32 for numerical stability under AMP
        logits_f32 = logits.float()

        # Use log_softmax for numerical stability (instead of separate softmax + log)
        log_probs = F.log_softmax(logits_f32, dim=1)  # [B, C, H, W]
        probs = log_probs.exp()  # [B, C, H, W]

        # Create valid mask and clean targets
        valid = (targets != self.ignore_index)
        targets_clean = targets.clone()
        targets_clean[~valid] = 0

        # Gather p_t and log(p_t) for the target class
        targets_flat = targets_clean.unsqueeze(1)  # [B, 1, H, W]
        log_p_t = log_probs.gather(1, targets_flat).squeeze(1)  # [B, H, W]
        p_t = probs.gather(1, targets_flat).squeeze(1)  # [B, H, W]

        # Clamp for safety
        p_t = p_t.clamp(min=1e-6, max=1.0 - 1e-6)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Focal loss: -alpha * (1-p_t)^gamma * log(p_t)
        loss = -focal_weight * log_p_t

        # Apply per-class alpha weights
        if self.alpha is not None:
            alpha = self.alpha.to(loss.device)
            alpha_t = alpha.gather(0, targets_clean.view(-1)).view_as(loss)
            loss = alpha_t * loss

        # Mask out ignore pixels
        loss = loss[valid]

        return loss.mean() if loss.numel() > 0 else loss.sum()


class DiceLoss(nn.Module):
    """Dice Loss for handling class imbalance.
    
    Numerically stable implementation that works under AMP (float16).
    """

    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W]
            targets: [B, H, W]
        """
        num_classes = logits.shape[1]

        # Cast to float32 for stability under AMP
        probs = F.softmax(logits.float(), dim=1)

        # Create valid mask
        valid = (targets != self.ignore_index)
        targets_clean = targets.clone()
        targets_clean[~valid] = 0

        # One-hot encoding
        targets_one_hot = F.one_hot(
            targets_clean, num_classes
        ).permute(0, 3, 1, 2).float()

        # Mask invalid pixels
        valid_mask = valid.unsqueeze(1).float()
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        # Compute per-class Dice
        dims = (0, 2, 3)  # sum over batch, height, width
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = probs.sum(dims) + targets_one_hot.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Average over classes (skip background for better performance)
        loss = 1.0 - dice[1:].mean()  # skip class 0 (background)
        return loss


class LovaszSoftmax(nn.Module):
    """Lovász-Softmax Loss.

    Directly optimizes the mean IoU via the Lovász extension of
    submodular set functions. More effective than Dice for IoU
    optimization, especially for minority classes.

    Reference: Berman et al., "The Lovász-Softmax loss: A tractable
    surrogate for the optimization of the intersection-over-union
    measure in neural networks", CVPR 2018.
    """

    def __init__(self, ignore_index=255, per_image=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    @staticmethod
    def _lovasz_grad(gt_sorted):
        """Compute gradient of the Lovász extension w.r.t sorted errors."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def _lovasz_softmax_flat(self, probs, labels, classes='present'):
        """Multi-class Lovász-Softmax loss on flattened predictions."""
        if probs.numel() == 0:
            return probs * 0.0
        C = probs.shape[0]
        losses = []
        for c in range(C):
            fg = (labels == c).float()
            if fg.sum() == 0 and classes == 'present':
                continue
            if C == 1:
                fg_class = 1.0 - probs[:, 0]
            else:
                fg_class = 1.0 - probs[c]
            errors = (fg - fg_class).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            fg_sorted = fg[perm]
            grad = self._lovasz_grad(fg_sorted)
            losses.append(torch.dot(errors_sorted, grad))
        if not losses:
            return torch.tensor(0.0, device=probs.device, requires_grad=True)
        return torch.stack(losses).mean()

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W]
            targets: [B, H, W]
        """
        logits_f32 = logits.float()
        probs = F.softmax(logits_f32, dim=1)
        B, C, H, W = probs.shape

        if self.per_image:
            losses = []
            for b in range(B):
                valid = targets[b] != self.ignore_index
                if valid.sum() == 0:
                    continue
                p = probs[b, :, valid]   # [C, N]
                t = targets[b, valid]     # [N]
                losses.append(self._lovasz_softmax_flat(p, t))
            if not losses:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            return torch.stack(losses).mean()
        else:
            valid = targets != self.ignore_index
            p = probs.permute(0, 2, 3, 1).reshape(-1, C)  # [BHW, C]
            t = targets.reshape(-1)                         # [BHW]
            v = valid.reshape(-1)
            p = p[v].t()  # [C, N]
            t = t[v]      # [N]
            return self._lovasz_softmax_flat(p, t)


class BoundaryLoss(nn.Module):
    """Boundary Supervision Loss.

    Extracts class boundaries from GT mask using morphological edge detection
    (max-pool dilation), then computes weighted BCE on boundary pixels.
    Forces the model to pay extra attention to class transition regions.
    """

    def __init__(self, num_classes=4, kernel_size=3, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.ignore_index = ignore_index

    def _extract_boundary(self, masks):
        """Extract boundary from segmentation masks using max-pool dilation.

        Args:
            masks: [B, H, W] integer class labels
        Returns:
            boundary: [B, 1, H, W] binary boundary map (float)
        """
        # One-hot encode
        B, H, W = masks.shape
        one_hot = F.one_hot(masks.clamp(0, self.num_classes - 1),
                            self.num_classes).permute(0, 3, 1, 2).float()
        # Dilate each class channel
        pad = self.kernel_size // 2
        dilated = F.max_pool2d(one_hot, self.kernel_size, stride=1, padding=pad)
        # Boundary = dilated != original (class transitions)
        diff = (dilated - one_hot).abs().sum(dim=1, keepdim=True)
        boundary = (diff > 0).float()
        return boundary

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W]
            targets: [B, H, W]
        """
        # Extract GT boundary
        boundary_gt = self._extract_boundary(targets)  # [B, 1, H, W]

        # Predict boundary: max prob - second max prob < threshold → boundary
        probs = F.softmax(logits.float(), dim=1)
        top2 = probs.topk(2, dim=1).values
        boundary_pred = 1.0 - (top2[:, 0:1] - top2[:, 1:2])  # Low confidence → boundary

        # BCE loss (boundary_pred is already in [0,1] from softmax, so use
        # binary_cross_entropy, NOT binary_cross_entropy_with_logits)
        boundary_pred = boundary_pred.clamp(1e-6, 1 - 1e-6)
        # Manual pos_weight: weight boundary pixels more since they're rare
        weight = torch.where(boundary_gt > 0.5,
                             torch.tensor(5.0, device=logits.device),
                             torch.tensor(1.0, device=logits.device))
        loss = F.binary_cross_entropy(boundary_pred, boundary_gt, weight=weight)
        return loss


class HybridLoss(nn.Module):
    """Hybrid Loss: Focal + Dice (+ optional Boundary).

    - Focal: handles easy/hard example mining with (1-p_t)^γ
    - Dice: region-level overlap metric, handles class imbalance
    - Boundary (optional): extra supervision on class transition regions

    Total = w_focal * Focal + w_dice * Dice + w_boundary * Boundary
    """

    def __init__(self, num_classes=4, focal_alpha=None, focal_gamma=2.0,
                 loss_weights=(1.0, 1.0), ignore_index=255,
                 boundary_weight=0.0):
        super().__init__()
        self.focal = FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma,
            ignore_index=ignore_index
        )
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.w_focal = loss_weights[0]
        self.w_dice = loss_weights[1] if len(loss_weights) > 1 else 1.0
        self.w_boundary = boundary_weight
        if boundary_weight > 0:
            self.boundary = BoundaryLoss(num_classes=num_classes, ignore_index=ignore_index)
        else:
            self.boundary = None

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        total = self.w_focal * focal_loss + self.w_dice * dice_loss
        if self.boundary is not None:
            total = total + self.w_boundary * self.boundary(logits, targets)
        return total, focal_loss, dice_loss

