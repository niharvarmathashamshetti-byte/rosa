# Exact Loss Implementation & Parameter Verification

## 1. Exact Loss Function Used in Experiment 002

In src/model/losses.py:
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

In scripts/train_v2.py (lines 94-99):
class_weights = torch.tensor([0.1, 1.0, 4.0, 1.0, 8.0, 8.0], dtype=torch.float32).to(device)
criterion = DiceCELoss(dice_weight=0.5, ce_weight=0.5, class_weights=class_weights)

---

## 2. Confirmation of bg_weight=0.1 Application

- bg_weight=0.1 was passed as index 0 of class_weights to PyTorch nn.CrossEntropyLoss(weight=class_weights).
- It was ONLY applied to the Cross-Entropy component.
- In SoftDiceLoss, background is completely excluded (include_background=False), meaning SoftDiceLoss only computes intersection and cardinality across foreground channels 1..5.
- Impact: The background class received a 10x smaller penalty in Cross-Entropy relative to bone (weight=1.0) and an 80x smaller penalty relative to rare cartilage (weight=8.0). Because Dice excluded background, the model suffered virtually zero loss when expanding predictions into background space.

---

## 3. Exact Class Weights in Experiment 002

- Class 0 (Background): 0.1 (Cross-Entropy), Excluded in Dice
- Class 1 (Femur): 1.0 (Cross-Entropy), Included in Dice
- Class 2 (Femoral Cartilage): 4.0 (Cross-Entropy), Included in Dice
- Class 3 (Tibia): 1.0 (Cross-Entropy), Included in Dice
- Class 4 (Medial Tibial Cartilage): 8.0 (Cross-Entropy), Included in Dice
- Class 5 (Lateral Tibial Cartilage): 8.0 (Cross-Entropy), Included in Dice

---

## 4. Decomposition of Dice and Cross-Entropy Components

- Total Loss = 0.5 * loss_dice + 0.5 * loss_ce
- SoftDiceLoss computes:
  DiceScore_c = (2 * sum(p_c * y_c) + eps) / (sum(p_c) + sum(y_c) + eps) for c in [1..5]
  Loss_Dice = 1 - mean(DiceScore_c)
  Symmetrical penalty on FP and FN.
- CrossEntropyLoss computes standard multi-class cross-entropy where true background voxels had loss scaled by 0.1.

---

## 5. Inspection of Tversky Loss Implementation & Formulas

From monai.losses.TverskyLoss:
  tp, fp, fn = compute_tp_fp_fn(input, target, reduce_axis, 1, self.soft_label, False)
  fp *= self.alpha
  fn *= self.beta
  numerator = tp + self.smooth_nr
  denominator = tp + fp + fn + self.smooth_dr
  score = 1.0 - numerator / denominator

Mathematical Formula:
  TverskyIndex = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
  Loss_Tversky = 1 - TverskyIndex

---

## 6. Verification of Parameter Roles (Alpha vs Beta)

In MONAI TverskyLoss implementation:
- alpha directly scales FALSE POSITIVES (FP).
- beta directly scales FALSE NEGATIVES (FN).

---

## 7. Correct Parameter Configuration for Experiment 003

To directly suppress the massive False Positive over-segmentation (87.97M femur FP voxels, 25.61M tibia FP voxels) observed in Experiment 002:

1. Tversky Parameters:
   - alpha = 0.7 (Heavy penalty on False Positives)
   - beta = 0.3 (Lower penalty on False Negatives)
   (Setting alpha > beta forces the optimizer to shrink predicted boundaries and eliminate the background halo).
2. Cross-Entropy Background Weight:
   - bg_weight = 1.0 (Restore background penalty to unity).
   - Class weights: [1.0, 1.0, 2.0, 1.0, 3.0, 3.0]
3. Loss Combination:
   - Loss = 0.5 * TverskyLoss(alpha=0.7, beta=0.3, include_background=False, softmax=True) + 0.5 * CrossEntropyLoss(weight=class_weights)
