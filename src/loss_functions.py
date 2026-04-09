"""
loss_functions.py
=================
Loss function yang lebih robust untuk face anti-spoofing dengan 6 kelas.

Masalah pada loss CrossEntropy standar:
  - Terlalu percaya diri (overconfident) → mudah salah di data test
  - Tidak memperhatikan kelas sulit (fake_screen sering salah)

Solusi di file ini:
  1. LabelSmoothingFocalLoss  : gabungan focal + label smoothing (TERBAIK)
  2. MixupCriterion           : loss untuk Mixup augmentation
  3. ClassWeightedFocalLoss   : focal loss dengan class weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────
# 1. Label Smoothing Focal Loss (Rekomendasi Utama)
# ─────────────────────────────────────────────────────
class LabelSmoothingFocalLoss(nn.Module):
    """
    Gabungan Label Smoothing + Focal Loss.

    Label Smoothing (smoothing=0.1):
      - Cegah model terlalu "overconfident"
      - Meningkatkan generalisasi ke domain baru (test Kaggle)
      - Target berubah: 1.0 → 0.9, 0.0 → 0.0111 (utk 6 kelas)

    Focal Loss (gamma=2.0):
      - Fokuskan training pada contoh "sulit" (fake_screen yang mirip realperson)
      - Down-weight contoh mudah yang sudah benar
      - alpha mengontrol class weighting

    Args:
        num_classes   : jumlah kelas (6 untuk dataset Anda)
        smoothing     : label smoothing factor (0.05–0.15 direkomendasikan)
        gamma         : focal loss exponent (1.5–3.0; 2.0 adalah default baik)
        alpha         : weight per kelas (None = uniform)
        reduction     : 'mean' atau 'sum'
    """

    def __init__(
        self,
        num_classes: int = 6,
        smoothing: float = 0.1,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        # Epsilon untuk numerik stability
        self.eps = 1e-8

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, num_classes) raw output dari model (sebelum softmax)
            targets : (B,) class indices
        Returns:
            scalar loss
        """
        B, C = logits.shape
        device = logits.device

        # ── Buat soft target dengan label smoothing ──
        # one-hot + smoothing
        smooth_val = self.smoothing / (C - 1)
        soft_targets = torch.full((B, C), smooth_val, device=device)
        soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # ── Hitung log probabilities ──
        log_prob = F.log_softmax(logits, dim=1)  # (B, C)
        prob     = torch.exp(log_prob)            # (B, C)

        # ── Focal weight: (1 - p_t)^gamma ──
        # p_t adalah probability untuk kelas yang benar
        p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        focal_weight = (1.0 - p_t + self.eps) ** self.gamma    # (B,)

        # ── Cross entropy dengan soft targets ──
        loss_per_sample = -(soft_targets * log_prob).sum(dim=1)  # (B,)

        # ── Terapkan focal weight ──
        loss_per_sample = focal_weight * loss_per_sample

        # ── Terapkan class alpha weight (opsional) ──
        if self.alpha is not None:
            alpha_t = self.alpha.to(device)[targets]  # (B,)
            loss_per_sample = alpha_t * loss_per_sample

        # ── Reduction ──
        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:
            return loss_per_sample


# ─────────────────────────────────────────────────────
# 2. Mixup Criterion
# ─────────────────────────────────────────────────────
class MixupCriterion(nn.Module):
    """
    Loss untuk Mixup augmentation.
    Mixup mencampur dua gambar: x = λ*x_a + (1-λ)*x_b
    Loss = λ*L(pred, y_a) + (1-λ)*L(pred, y_b)

    Cara pakai dengan training loop:
        # Di dalam DataLoader / collate:
        lam = np.random.beta(alpha, alpha)
        x_mix = lam * x_a + (1 - lam) * x_b

        # Di training step:
        criterion = MixupCriterion(base_criterion)
        loss = criterion(logits, y_a, y_b, lam)
    """

    def __init__(self, base_criterion: nn.Module):
        super().__init__()
        self.base = base_criterion

    def forward(
        self,
        logits: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        return lam * self.base(logits, y_a) + (1 - lam) * self.base(logits, y_b)


# ─────────────────────────────────────────────────────
# 3. Class-Weighted Focal Loss (alternatif)
# ─────────────────────────────────────────────────────
class ClassWeightedFocalLoss(nn.Module):
    """
    Focal loss dengan class weights eksplisit.
    Gunakan ini jika distribusi kelas sangat imbalanced.

    Args:
        class_weights : tensor of shape (num_classes,) — bisa dihitung dari
                        distribusi dataset, misalnya 1/freq
        gamma         : focal exponent
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        prob     = torch.exp(log_prob)

        # Ambil prob dan log_prob untuk kelas target
        p_t   = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        lp_t  = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_w = (1.0 - p_t) ** self.gamma

        # Class weight
        class_w = self.class_weights[targets]

        loss = -class_w * focal_w * lp_t
        return loss.mean()


# ─────────────────────────────────────────────────────
# 4. Helper: Hitung class weights dari dataset
# ─────────────────────────────────────────────────────
def compute_class_weights(labels: list, num_classes: int = 6) -> torch.Tensor:
    """
    Hitung class weights inversely proportional terhadap frekuensi.

    Args:
        labels      : list/array of integer labels dari seluruh training set
        num_classes : jumlah kelas

    Returns:
        torch.Tensor of shape (num_classes,) — normalized weights

    Contoh:
        from loss_functions import compute_class_weights
        all_labels = [dataset[i][1] for i in range(len(dataset))]
        weights = compute_class_weights(all_labels, num_classes=6)
        criterion = ClassWeightedFocalLoss(weights)
    """
    import numpy as np
    labels_arr = np.array(labels)
    counts = np.bincount(labels_arr, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1)  # hindari division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalisasi ke jumlah kelas
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────
# 5. Konfigurasi yang Direkomendasikan
# ─────────────────────────────────────────────────────
def get_recommended_criterion(
    use_class_weights: bool = False,
    class_weights: Optional[torch.Tensor] = None,
    device: str = "cuda"
) -> nn.Module:
    """
    Kembalikan loss function yang direkomendasikan untuk proyek ini.

    Catatan penting untuk 6 kelas Anda:
      - realperson       : kelas sulit (banyak false negative)
      - fake_screen      : paling sering salah → perlu focal loss
      - fake_printed     : biasanya mudah (ada tekstur jelas)
      - fake_mask        : menengah
      - fake_mannequin   : menengah
      - fake_unknown     : paling sulit (definisi ambigu)

    Args:
        use_class_weights : True jika distribusi kelas imbalanced
        class_weights     : tensor weights (diperlukan jika use_class_weights=True)
        device            : 'cuda' atau 'cpu'
    """
    if use_class_weights and class_weights is not None:
        alpha = class_weights.to(device)
    else:
        alpha = None

    criterion = LabelSmoothingFocalLoss(
        num_classes=6,
        smoothing=0.1,    # Tidak terlalu aggressive, tidak terlalu konservatif
        gamma=2.0,        # Standard focal loss
        alpha=alpha,
        reduction="mean"
    ).to(device)

    return criterion


# ─────────────────────────────────────────────────────
# 6. Mixup helper function (untuk dipakai di training loop)
# ─────────────────────────────────────────────────────
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """
    Lakukan Mixup pada batch x, y.

    Args:
        x     : (B, C, H, W) batch gambar
        y     : (B,) label integer
        alpha : parameter distribusi Beta (0.2–0.8 biasanya bagus)

    Returns:
        x_mixed, y_a, y_b, lam
    """
    import numpy as np
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    B = x.size(0)
    index = torch.randperm(B, device=x.device)

    x_mixed = lam * x + (1 - lam) * x[index]
    y_a = y
    y_b = y[index]

    return x_mixed, y_a, y_b, lam


# ─────────────────────────────────────────────────────
# 7. Contoh penggunaan di training loop
# ─────────────────────────────────────────────────────
TRAINING_LOOP_EXAMPLE = """
# ── Di awal training ──
from loss_functions import (
    get_recommended_criterion, MixupCriterion, mixup_data
)

base_criterion = get_recommended_criterion(device=device)
mixup_criterion = MixupCriterion(base_criterion)
USE_MIXUP = True  # Toggle mixup on/off

# ── Di dalam training loop ──
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()

    with torch.amp.autocast("cuda"):
        if USE_MIXUP and random.random() < 0.5:
            images_mix, y_a, y_b, lam = mixup_data(images, labels, alpha=0.4)
            logits = model(images_mix)
            loss = mixup_criterion(logits, y_a, y_b, lam)
        else:
            logits = model(images)
            loss = base_criterion(logits, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
"""

if __name__ == "__main__":
    # Sanity check
    B, C = 8, 6
    logits  = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))

    crit = LabelSmoothingFocalLoss(num_classes=C, smoothing=0.1, gamma=2.0)
    loss = crit(logits, targets)
    print(f"LabelSmoothingFocalLoss: {loss.item():.4f}")

    weights = compute_class_weights(targets.tolist(), num_classes=C)
    print(f"Class weights: {weights}")

    x = torch.randn(B, 3, 224, 224)
    x_mix, ya, yb, lam = mixup_data(x, targets)
    print(f"Mixup lambda: {lam:.4f}")
    print("loss_functions.py OK")
