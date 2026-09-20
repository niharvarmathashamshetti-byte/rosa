import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, include_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, D, H, W)
        # targets: (B, D, H, W) integer labels
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        
        target_one_hot = F.one_hot(targets.long(), num_classes=num_classes) # (B, D, H, W, C)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()     # (B, C, D, H, W)
        
        start_class = 0 if self.include_background else 1
        
        intersection = torch.sum(probs[:, start_class:] * target_one_hot[:, start_class:], dim=(0, 2, 3, 4))
        cardinality = torch.sum(probs[:, start_class:] + target_one_hot[:, start_class:], dim=(0, 2, 3, 4))
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - torch.mean(dice_score)
        return dice_loss

class DiceCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5, class_weights: torch.Tensor = None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = SoftDiceLoss(include_background=False)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_dice = self.dice(logits, targets)
        loss_ce = self.ce(logits, targets.long())
        return self.dice_weight * loss_dice + self.ce_weight * loss_ce
