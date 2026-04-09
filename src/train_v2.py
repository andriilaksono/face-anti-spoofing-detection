"""
train_v2.py
===========
Script training yang diperbaiki dengan semua peningkatan akurasi:

  1. Augmentasi domain-aware (augmentation_v2.py)
  2. LabelSmoothingFocalLoss (loss_functions.py)
  3. Cosine Annealing dengan Warm Restarts (scheduler lebih baik)
  4. Gradient Clipping (stabilitas training)
  5. Early Stopping + checkpoint berdasarkan val accuracy
  6. Layer-wise Learning Rate Decay (backbone vs head berbeda LR)
  7. Mixup augmentation

Cara pakai:
    python train_v2.py --data_dir data/train_cropped \
                       --model convnext \
                       --epochs 50 \
                       --batch_size 32
"""

import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import timm
from tqdm import tqdm
import json

from src.augmentation_v2 import get_train_transforms, get_val_transforms
from loss_functions import (
    get_recommended_criterion, MixupCriterion, mixup_data, compute_class_weights
)


# ─────────────────────────────────────────────────────
# Konfigurasi Global
# ─────────────────────────────────────────────────────
CLASS_NAMES = [
    "realperson", "fake_screen", "fake_printed",
    "fake_mask", "fake_mannequin", "fake_unknown",
]
NUM_CLASSES = len(CLASS_NAMES)


def set_seed(seed: int = 42):
    """Pastikan reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────
class FaceDataset(torch.utils.data.Dataset):
    """
    Dataset untuk gambar wajah yang sudah di-crop (dari train_cropped/).
    Struktur folder:
        train_cropped/
            realperson/
                img001.jpg
                ...
            fake_screen/
                img002.jpg
                ...
            ...
    """

    def __init__(self, data_dir: str, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

        data_dir = Path(data_dir)
        for class_name in CLASS_NAMES:
            class_dir = data_dir / class_name
            if not class_dir.exists():
                print(f"[WARN] Folder tidak ditemukan: {class_dir}")
                continue
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

        print(f"[Dataset] Total: {len(self.samples)} gambar")
        # Cetak distribusi kelas
        from collections import Counter
        dist = Counter(label for _, label in self.samples)
        for cls_name, idx in self.class_to_idx.items():
            print(f"  {cls_name:20s}: {dist.get(idx, 0):5d}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_all_labels(self):
        return [label for _, label in self.samples]


# ─────────────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────────────
def build_model(model_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """
    Bangun model dengan timm.
    Backbone: ConvNeXt-Tiny atau EfficientNet-B3.
    """
    arch_map = {
        "convnext": "convnext_tiny",
        "efficientnet": "efficientnet_b3",
    }
    arch = arch_map.get(model_name.lower(), model_name)
    model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
    print(f"[Model] {arch} | params: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ─────────────────────────────────────────────────────
# Optimizer dengan Layer-wise LR Decay
# ─────────────────────────────────────────────────────
def build_optimizer(model: nn.Module, base_lr: float = 3e-4, backbone_lr_mult: float = 0.1):
    """
    Layer-wise LR: backbone belajar lebih lambat (LR kecil),
    head classifier belajar lebih cepat (LR besar).
    Ini mencegah catastrophic forgetting pada pretrained weights.

    Args:
        base_lr          : LR untuk head/classifier layer
        backbone_lr_mult : multiplier untuk backbone (0.1 = 10x lebih kecil)
    """
    # Pisahkan parameter backbone dan head
    head_params     = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # ConvNeXt: head = 'head.*', EfficientNet: head = 'classifier.*'
        if any(h in name for h in ["head", "classifier", "fc"]):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
        {"params": head_params,     "lr": base_lr},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=1e-4,
        eps=1e-8
    )
    print(f"[Optimizer] AdamW | backbone LR: {base_lr * backbone_lr_mult:.1e} | head LR: {base_lr:.1e}")
    return optimizer


# ─────────────────────────────────────────────────────
# Training & Validation Step
# ─────────────────────────────────────────────────────
def train_one_epoch(
    model, loader, criterion, mixup_criterion,
    optimizer, scaler, scheduler_warmup,
    device, use_mixup=True, mixup_alpha=0.4,
    grad_clip=1.0
):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            if use_mixup and random.random() < 0.5:
                images_mix, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                logits = model(images_mix)
                loss = mixup_criterion(logits, y_a, y_b, lam)
                # Accuracy estimation (approximate)
                preds = logits.argmax(dim=1)
                correct = int((lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()).sum())
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()

        scaler.scale(loss).backward()

        # Gradient clipping untuk stabilitas
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if scheduler_warmup is not None:
            scheduler_warmup.step()

        total_loss    += loss.item() * images.size(0)
        total_correct += correct
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(images)
            loss   = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        total_loss    += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc, all_preds, all_labels


# ─────────────────────────────────────────────────────
# Per-class accuracy report
# ─────────────────────────────────────────────────────
def print_class_report(preds, labels):
    from collections import defaultdict
    correct_per_class = defaultdict(int)
    total_per_class   = defaultdict(int)

    for p, l in zip(preds, labels):
        total_per_class[l] += 1
        if p == l:
            correct_per_class[l] += 1

    print("  ┌─────────────────────────┬────────┬────────┐")
    print("  │ Kelas                   │ Total  │  Acc   │")
    print("  ├─────────────────────────┼────────┼────────┤")
    for i, name in enumerate(CLASS_NAMES):
        total   = total_per_class.get(i, 0)
        correct = correct_per_class.get(i, 0)
        acc_str = f"{correct/total*100:.1f}%" if total > 0 else "N/A"
        print(f"  │ {name:23s} │ {total:6d} │ {acc_str:6s} │")
    print("  └─────────────────────────┴────────┴────────┘")


# ─────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────
def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}")

    # ── Dataset ──
    full_dataset = FaceDataset(args.data_dir)
    all_labels   = full_dataset.get_all_labels()
    n_total      = len(full_dataset)
    n_val        = int(n_total * args.val_split)
    n_train      = n_total - n_val

    train_base, val_base = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    model_name = args.model.lower()

    # Terapkan transform berbeda ke train dan val
    class TransformSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            return self.transform(img), label

    train_dataset = TransformSubset(train_base, get_train_transforms(model_name))
    val_dataset   = TransformSubset(val_base,   get_val_transforms(model_name))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"[Dataset] Train: {n_train} | Val: {n_val}")

    # ── Model ──
    model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=True).to(device)

    # ── Loss ──
    # Hitung class weights dari training set
    train_labels = [all_labels[i] for i in train_base.indices]
    class_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)

    criterion       = get_recommended_criterion(
        use_class_weights=True,
        class_weights=class_weights,
        device=device
    )
    mixup_criterion = MixupCriterion(criterion)

    # ── Optimizer ──
    optimizer = build_optimizer(model, base_lr=args.lr, backbone_lr_mult=0.1)

    # ── Scheduler: Cosine Annealing with Warm Restarts ──
    steps_per_epoch = len(train_loader)
    total_steps     = args.epochs * steps_per_epoch

    # Warmup: 5% pertama dari training
    warmup_steps = int(total_steps * 0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=steps_per_epoch * 10, T_mult=1, eta_min=1e-6
    )

    # ── AMP Scaler ──
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # ── Training ──
    best_val_acc  = 0.0
    patience_cnt  = 0
    os.makedirs(args.output_dir, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n[Train] Mulai training {args.epochs} epoch...\n")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch:3d}/{args.epochs}")

        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, mixup_criterion,
            optimizer, scaler, scheduler,
            device, use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha,
            grad_clip=args.grad_clip
        )

        # Validation
        val_loss, val_acc, val_preds, val_labels_list = validate(
            model, val_loader, criterion, device
        )

        # Cetak per-class accuracy setiap 5 epoch
        if epoch % 5 == 0:
            print_class_report(val_preds, val_labels_list)

        # Simpan history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Loss: train={train_loss:.4f} val={val_loss:.4f} | "
            f"Acc: train={train_acc*100:.2f}% val={val_acc*100:.2f}%"
        )

        # Simpan model terbaik
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            ckpt_path = os.path.join(args.output_dir, f"{model_name}_best.pth")
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_acc":          val_acc,
                "val_loss":         val_loss,
                "args":             vars(args),
            }, ckpt_path)
            print(f"  ✓ Best model saved (val_acc={val_acc*100:.2f}%)")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n[EarlyStopping] Tidak ada peningkatan selama {args.patience} epoch. Stop.")
                break

    # Simpan history
    history_path = os.path.join(args.output_dir, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Train] Selesai! Best val acc: {best_val_acc*100:.2f}%")
    print(f"[Train] Checkpoint: {os.path.join(args.output_dir, f'{model_name}_best.pth')}")


# ─────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Face Anti-Spoofing Training v2")
    parser.add_argument("--data_dir",    type=str, default="data/train_cropped",
                        help="Path ke folder train_cropped/")
    parser.add_argument("--model",       type=str, default="convnext",
                        choices=["convnext", "efficientnet"],
                        help="Arsitektur model yang akan dilatih")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--val_split",   type=float, default=0.15,
                        help="Proporsi data validasi (default: 15%)")
    parser.add_argument("--patience",    type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--use_mixup",   action="store_true", default=True)
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--grad_clip",   type=float, default=1.0)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output_dir",  type=str, default="checkpoints",
                        help="Direktori untuk menyimpan model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
