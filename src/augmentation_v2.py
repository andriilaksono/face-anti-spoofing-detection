"""
augmentation_v2.py
==================
Augmentasi domain-aware untuk mengatasi gap 92% (lokal) → 73% (Kaggle).
Fokus: simulasikan kondisi kamera buram, kompresi JPEG, layar HP beresolusi tinggi.

Cara pakai:
    from augmentation_v2 import get_train_transforms, get_val_transforms
    train_dataset = YourDataset(transform=get_train_transforms())
"""

import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import io


# ─────────────────────────────────────────────
# 1. Custom transform: simulasi kompresi JPEG
# ─────────────────────────────────────────────
class RandomJPEGCompression:
    """
    Simulasi artefak JPEG seperti pada foto layar HP/print.
    Quality rendah = lebih banyak artefak blok → model belajar tetap robust.
    """
    def __init__(self, quality_range=(40, 95)):
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


# ─────────────────────────────────────────────
# 2. Custom transform: simulasi blur kamera
# ─────────────────────────────────────────────
class RandomCameraBlur:
    """
    Simulasi blur akibat gerakan tangan atau fokus kamera buruk.
    Ini adalah penyebab utama domain gap pada data test Kaggle.
    """
    def __init__(self, blur_prob=0.4, kernel_range=(3, 7)):
        self.blur_prob = blur_prob
        self.kernel_range = kernel_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.blur_prob:
            # Pilih kernel ganjil secara acak
            k = random.choice([k for k in range(
                self.kernel_range[0], self.kernel_range[1] + 1, 2
            )])
            sigma = random.uniform(0.5, 2.5)
            return TF.gaussian_blur(img, kernel_size=k, sigma=sigma)
        return img


# ─────────────────────────────────────────────
# 3. Custom transform: CLAHE-like histogram equalization
# ─────────────────────────────────────────────
class RandomHistogramEqualization:
    """
    Simulasi variasi pencahayaan ekstrem (terlalu gelap / terlalu terang).
    Membantu model robust terhadap kondisi lighting yang berbeda dari training.
    """
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.prob:
            # Equalize histogram untuk simulasi kamera dengan auto-exposure berbeda
            return TF.equalize(img)
        return img


# ─────────────────────────────────────────────
# 4. Custom transform: simulasi layar HP
# ─────────────────────────────────────────────
class RandomScreenSimulation:
    """
    Simulasi pola Moiré dan pixel grid dari layar HP/monitor.
    Ini yang membuat fake_screen sering salah klasifikasi sebagai realperson.
    Tambahkan noise halus + slight color shift biru/hijau (LCD color cast).
    """
    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.prob:
            img_array = np.array(img, dtype=np.float32)

            # Slight blue/green cast (LCD screens)
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.05, 0, 255)
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.02, 0, 255)

            # Tambahkan noise Gaussian halus
            noise = np.random.normal(0, 3, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

            return Image.fromarray(img_array)
        return img


# ─────────────────────────────────────────────
# 5. GridMask — lebih effective dari Cutout untuk face data
# ─────────────────────────────────────────────
class GridMask:
    """
    Hapus region grid dari gambar untuk regularisasi.
    Lebih baik dari random erasing untuk data wajah karena
    menjaga proporsi spatial fitur wajah.
    """
    def __init__(self, prob=0.3, ratio=0.4, d_range=(60, 90)):
        self.prob = prob
        self.ratio = ratio
        self.d_range = d_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img

        w, h = img.size
        d = random.randint(*self.d_range)
        delta_x = random.randint(0, d)
        delta_y = random.randint(0, d)
        l = int(d * self.ratio + 0.5)

        mask = np.ones((h, w), dtype=np.float32)
        for i in range(-1, h // d + 1):
            for j in range(-1, w // d + 1):
                x = d * j - delta_x
                y = d * i - delta_y
                x1, x2 = max(x, 0), min(x + l, w)
                y1, y2 = max(y, 0), min(y + l, h)
                if x1 < x2 and y1 < y2:
                    mask[y1:y2, x1:x2] = 0

        img_array = np.array(img)
        mask_3c = np.stack([mask] * 3, axis=2)
        img_array = (img_array * mask_3c).astype(np.uint8)
        return Image.fromarray(img_array)


# ─────────────────────────────────────────────
# 6. Pipeline lengkap
# ─────────────────────────────────────────────

# Ukuran input masing-masing model
CONVNEXT_SIZE = 224  # ConvNeXt-Tiny
EFFICIENTNET_SIZE = 300  # EfficientNet-B3

# Nilai normalisasi ImageNet (sama untuk keduanya karena pretrained)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(model_name: str = "convnext"):
    """
    Training transform dengan augmentasi agresif untuk mengatasi domain gap.

    Args:
        model_name: 'convnext' (224px) atau 'efficientnet' (300px)

    Returns:
        torchvision.transforms.Compose
    """
    size = CONVNEXT_SIZE if model_name == "convnext" else EFFICIENTNET_SIZE

    return T.Compose([
        # ── Step 1: Resize & random crop untuk scale invariance ──
        T.Resize((size + 32, size + 32)),
        T.RandomCrop(size),
        T.RandomHorizontalFlip(p=0.5),

        # ── Step 2: Simulasi kondisi kamera nyata (domain gap fix) ──
        RandomCameraBlur(blur_prob=0.4, kernel_range=(3, 7)),
        RandomHistogramEqualization(prob=0.3),
        RandomScreenSimulation(prob=0.15),

        # ── Step 3: Color augmentation agresif ──
        T.ColorJitter(
            brightness=0.4,   # variasi gelap/terang
            contrast=0.4,     # variasi kontras
            saturation=0.3,   # variasi saturasi warna
            hue=0.1           # slight hue shift
        ),
        T.RandomGrayscale(p=0.05),

        # ── Step 4: Simulasi kompresi (penting untuk fake_screen!) ──
        RandomJPEGCompression(quality_range=(50, 95)),

        # ── Step 5: Regularisasi spasial ──
        GridMask(prob=0.3),
        T.RandomRotation(degrees=10),
        T.RandomPerspective(distortion_scale=0.2, p=0.3),

        # ── Step 6: Convert dan normalize ──
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # ── Step 7: Random erasing sebagai final regularization ──
        T.RandomErasing(
            p=0.2,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value='random'
        ),
    ])


def get_val_transforms(model_name: str = "convnext"):
    """
    Validation transform — TIDAK ada augmentasi acak, hanya resize + normalize.

    Args:
        model_name: 'convnext' (224px) atau 'efficientnet' (300px)
    """
    size = CONVNEXT_SIZE if model_name == "convnext" else EFFICIENTNET_SIZE

    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transforms(model_name: str = "convnext"):
    """
    Kumpulan transform untuk Test Time Augmentation (TTA).
    Setiap elemen diaplikasikan secara terpisah saat inference.

    Args:
        model_name: 'convnext' atau 'efficientnet'

    Returns:
        list of torchvision.transforms.Compose
    """
    size = CONVNEXT_SIZE if model_name == "convnext" else EFFICIENTNET_SIZE

    base = [
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    tta_list = [
        # Original
        T.Compose(base),

        # Horizontal flip
        T.Compose([T.Resize((size, size)), T.RandomHorizontalFlip(p=1.0),
                   T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),

        # Slight brightness boost (simulasi layar terang)
        T.Compose([T.Resize((size, size)),
                   T.ColorJitter(brightness=(1.2, 1.2)),
                   T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),

        # Slight brightness drop (simulasi pencahayaan redup)
        T.Compose([T.Resize((size, size)),
                   T.ColorJitter(brightness=(0.7, 0.7)),
                   T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),

        # Contrast boost
        T.Compose([T.Resize((size, size)),
                   T.ColorJitter(contrast=(1.3, 1.3)),
                   T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),

        # Slight blur (simulasi kamera buram)
        T.Compose([T.Resize((size, size)),
                   RandomCameraBlur(blur_prob=1.0, kernel_range=(3, 5)),
                   T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),

        # Center crop (zoom in sedikit)
        T.Compose([T.Resize((int(size * 1.15), int(size * 1.15))),
                   T.CenterCrop(size),
                   T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    ]

    return tta_list


if __name__ == "__main__":
    # Quick sanity check
    img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))

    train_tf = get_train_transforms("convnext")
    val_tf   = get_val_transforms("convnext")
    tta_list = get_tta_transforms("convnext")

    out_train = train_tf(img)
    out_val   = val_tf(img)

    print(f"Train output shape : {out_train.shape}")  # torch.Size([3, 224, 224])
    print(f"Val output shape   : {out_val.shape}")
    print(f"Jumlah TTA variant : {len(tta_list)}")
    print("augmentation_v2.py OK")
