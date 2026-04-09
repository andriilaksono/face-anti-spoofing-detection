"""
inference_v2.py
===============
Inference dengan TTA (Test Time Augmentation) yang diperluas dan
ensemble weight optimization.

Perbaikan utama vs kode lama:
  1. TTA 7 augmentasi (vs Multi-Zoom 3-level sebelumnya)
  2. Temperature Scaling untuk kalibrasi confidence
  3. Optimasi weight ConvNeXt vs EfficientNet via grid search
  4. Penanganan kasus face detection gagal yang lebih baik

Cara pakai:
    from inference_v2 import EnsemblePredictor
    predictor = EnsemblePredictor(
        convnext_path="model_convnext_best.pth",
        efficientnet_path="model_efficientnet_best.pth",
    )
    label, probs = predictor.predict("path/to/image.jpg")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List
import timm

try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False
    print("[WARN] facenet_pytorch tidak ditemukan, fallback ke center crop.")

from src.augmentation_v2 import get_tta_transforms, get_val_transforms


# ─────────────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────────────
CLASS_NAMES = [
    "realperson",
    "fake_screen",
    "fake_printed",
    "fake_mask",
    "fake_mannequin",
    "fake_unknown",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────
# 1. Temperature Scaling — kalibrasi confidence model
# ─────────────────────────────────────────────────────
class TemperatureScaler(nn.Module):
    """
    Post-hoc calibration: bagi logits dengan temperature T.
    T > 1: softmax lebih "flat" (lebih tidak percaya diri) → lebih generalize
    T < 1: softmax lebih "sharp" (lebih percaya diri)

    Cara cari T optimal:
        scaler = TemperatureScaler()
        scaler.calibrate(model, val_loader, device)
        # Lalu gunakan scaler(logits) saat inference
    """

    def __init__(self, temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def calibrate(self, model: nn.Module, val_loader, device: str = DEVICE):
        """
        Cari temperature T yang meminimalkan NLL pada validation set.
        Panggil ini SETELAH training selesai, gunakan validation loader.
        """
        from torch.optim import LBFGS

        model.eval()
        self.to(device)
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=50)

        logits_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                logits_list.append(logits.cpu())
                labels_list.append(labels)

        all_logits = torch.cat(logits_list).to(device)
        all_labels = torch.cat(labels_list).to(device)

        def eval_fn():
            optimizer.zero_grad()
            scaled = self.forward(all_logits)
            loss = F.cross_entropy(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        print(f"[TemperatureScaler] Optimal T = {self.temperature.item():.4f}")


# ─────────────────────────────────────────────────────
# 2. Face Detector dengan fallback yang lebih baik
# ─────────────────────────────────────────────────────
class FaceDetector:
    """
    Wrapper MTCNN dengan smart fallback bertingkat:
      1. Coba MTCNN dengan margin besar
      2. Jika gagal: coba MTCNN dengan threshold lebih rendah
      3. Jika tetap gagal: 5-crop (4 corner + center)
      4. Pilih crop yang paling "face-like" berdasarkan aspect ratio
    """

    def __init__(self, device: str = DEVICE):
        self.device = device
        if HAS_MTCNN:
            # MTCNN utama
            self.mtcnn_main = MTCNN(
                image_size=160,
                margin=44,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=device
            )
            # MTCNN fallback (threshold lebih permisif)
            self.mtcnn_fallback = MTCNN(
                image_size=160,
                margin=60,
                min_face_size=15,
                thresholds=[0.4, 0.5, 0.5],
                factor=0.709,
                post_process=False,
                device=device
            )

    def detect_and_crop(self, img: Image.Image, target_size: int = 224) -> Image.Image:
        """
        Deteksi dan crop wajah. Kembalikan PIL Image berukuran target_size x target_size.
        """
        w, h = img.size

        if HAS_MTCNN:
            # Percobaan 1: MTCNN normal
            face = self._try_mtcnn(self.mtcnn_main, img)
            if face is not None:
                return face.resize((target_size, target_size), Image.LANCZOS)

            # Percobaan 2: MTCNN fallback (threshold lebih rendah)
            face = self._try_mtcnn(self.mtcnn_fallback, img)
            if face is not None:
                return face.resize((target_size, target_size), Image.LANCZOS)

        # Percobaan 3: Smart center crop
        return self._smart_center_crop(img, target_size)

    def _try_mtcnn(self, mtcnn, img: Image.Image) -> Optional[Image.Image]:
        try:
            boxes, probs = mtcnn.detect(img)
            if boxes is not None and len(boxes) > 0:
                # Ambil box dengan confidence tertinggi
                best_idx = np.argmax(probs)
                box = boxes[best_idx].astype(int)
                x1, y1, x2, y2 = box
                w, h = img.size
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    return img.crop((x1, y1, x2, y2))
        except Exception:
            pass
        return None

    def _smart_center_crop(self, img: Image.Image, target_size: int) -> Image.Image:
        """
        Center crop dengan mempertimbangkan aspect ratio.
        Crop area yang paling mungkin mengandung wajah (tengah atas).
        """
        w, h = img.size
        # Wajah biasanya di 1/4 hingga 3/4 gambar secara vertikal
        # dan di tengah secara horizontal
        crop_size = min(w, h)
        cx = w // 2
        cy = int(h * 0.4)  # Sedikit ke atas dari tengah
        half = crop_size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        cropped = img.crop((x1, y1, x2, y2))
        return cropped.resize((target_size, target_size), Image.LANCZOS)


# ─────────────────────────────────────────────────────
# 3. Single Model Wrapper
# ─────────────────────────────────────────────────────
class SingleModelPredictor:
    """
    Wrapper untuk satu model (ConvNeXt atau EfficientNet).
    Melakukan TTA dan mengembalikan rata-rata probabilitas.
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        num_classes: int = 6,
        temperature: float = 1.3,
        device: str = DEVICE,
    ):
        self.model_name = model_name
        self.device = device
        self.temperature = temperature

        # ── Load model ──
        arch = "convnext_tiny" if "convnext" in model_name.lower() else "efficientnet_b3"
        self.model = timm.create_model(arch, pretrained=False, num_classes=num_classes)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Support berbagai format checkpoint
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device)
        self.model.eval()

        # ── TTA transforms ──
        base_name = "convnext" if "convnext" in model_name.lower() else "efficientnet"
        self.tta_transforms = get_tta_transforms(base_name)
        self.val_transform   = get_val_transforms(base_name)

        print(f"[{model_name}] loaded dari {checkpoint_path} | TTA: {len(self.tta_transforms)} augmentasi")

    @torch.no_grad()
    def predict_proba(
        self,
        img: Image.Image,
        use_tta: bool = True
    ) -> np.ndarray:
        """
        Kembalikan rata-rata softmax probability dari semua TTA variants.

        Args:
            img     : PIL Image (sudah di-crop wajah)
            use_tta : True = gunakan TTA, False = single forward pass

        Returns:
            np.ndarray shape (num_classes,) — probability per kelas
        """
        transforms = self.tta_transforms if use_tta else [self.val_transform]

        all_probs = []
        for tf in transforms:
            tensor = tf(img).unsqueeze(0).to(self.device)  # (1, C, H, W)
            with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                logits = self.model(tensor)
            # Temperature scaling
            logits = logits / self.temperature
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)

        return np.mean(all_probs, axis=0)  # shape: (num_classes,)


# ─────────────────────────────────────────────────────
# 4. Ensemble Predictor (utama)
# ─────────────────────────────────────────────────────
class EnsemblePredictor:
    """
    Ensemble ConvNeXt + EfficientNet dengan weight yang bisa dioptimasi.

    Cara cari weight optimal:
        predictor.optimize_weights(val_images, val_labels)

    Cara predict:
        label, probs = predictor.predict("path/to/image.jpg")
    """

    def __init__(
        self,
        convnext_path: str,
        efficientnet_path: str,
        convnext_weight: float = 0.65,
        efficientnet_weight: float = 0.35,
        num_classes: int = 6,
        temperature_convnext: float = 1.3,
        temperature_efficientnet: float = 1.2,
        device: str = DEVICE,
    ):
        self.device = device
        self.num_classes = num_classes
        self.w_cnx = convnext_weight
        self.w_eff = efficientnet_weight

        # Normalisasi weight
        total = self.w_cnx + self.w_eff
        self.w_cnx /= total
        self.w_eff /= total

        # ── Load kedua model ──
        self.convnext = SingleModelPredictor(
            "convnext", convnext_path,
            num_classes=num_classes,
            temperature=temperature_convnext,
            device=device
        )
        self.efficientnet = SingleModelPredictor(
            "efficientnet", efficientnet_path,
            num_classes=num_classes,
            temperature=temperature_efficientnet,
            device=device
        )

        # ── Face detector ──
        self.face_detector = FaceDetector(device=device)

        print(f"[Ensemble] ConvNeXt weight: {self.w_cnx:.2f} | EfficientNet weight: {self.w_eff:.2f}")

    def predict(
        self,
        image_path: str,
        use_tta: bool = True,
        return_all_probs: bool = False
    ) -> Tuple[str, np.ndarray]:
        """
        Prediksi satu gambar.

        Args:
            image_path      : path ke file gambar
            use_tta         : True = gunakan TTA (lebih akurat, lebih lambat)
            return_all_probs: True = kembalikan probs per kelas

        Returns:
            (predicted_class_name, probability_array)
        """
        img = Image.open(image_path).convert("RGB")

        # Deteksi dan crop wajah
        img_cropped = self.face_detector.detect_and_crop(img)

        # Prediksi dari kedua model
        probs_cnx = self.convnext.predict_proba(img_cropped, use_tta=use_tta)
        probs_eff = self.efficientnet.predict_proba(img_cropped, use_tta=use_tta)

        # Weighted ensemble
        probs_ensemble = self.w_cnx * probs_cnx + self.w_eff * probs_eff

        pred_idx  = np.argmax(probs_ensemble)
        pred_name = CLASS_NAMES[pred_idx]

        return pred_name, probs_ensemble

    def optimize_weights(
        self,
        val_image_paths: List[str],
        val_labels: List[int],
        search_step: float = 0.05,
        use_tta: bool = False  # False untuk kecepatan saat grid search
    ) -> Tuple[float, float]:
        """
        Grid search untuk menemukan weight ConvNeXt dan EfficientNet yang optimal.

        Args:
            val_image_paths : list path gambar validasi
            val_labels      : list label integer (0-5)
            search_step     : step grid search (default 0.05 = 20 kombinasi)
            use_tta         : True = gunakan TTA (lambat), False = single pass

        Returns:
            (best_w_convnext, best_w_efficientnet)
        """
        print(f"[EnsembleOptimizer] Grid search pada {len(val_image_paths)} gambar...")

        # Kumpulkan prediksi dari kedua model terlebih dahulu
        all_probs_cnx = []
        all_probs_eff = []

        for i, path in enumerate(val_image_paths):
            if i % 50 == 0:
                print(f"  Preprocessing {i}/{len(val_image_paths)}...")
            try:
                img = Image.open(path).convert("RGB")
                img_crop = self.face_detector.detect_and_crop(img)
                all_probs_cnx.append(self.convnext.predict_proba(img_crop, use_tta))
                all_probs_eff.append(self.efficientnet.predict_proba(img_crop, use_tta))
            except Exception as e:
                print(f"  [SKIP] {path}: {e}")
                all_probs_cnx.append(np.ones(self.num_classes) / self.num_classes)
                all_probs_eff.append(np.ones(self.num_classes) / self.num_classes)

        probs_cnx = np.array(all_probs_cnx)  # (N, 6)
        probs_eff = np.array(all_probs_eff)  # (N, 6)
        labels    = np.array(val_labels)      # (N,)

        # Grid search
        best_acc = 0.0
        best_w   = (0.65, 0.35)

        for w1 in np.arange(0.0, 1.0 + search_step, search_step):
            w2 = 1.0 - w1
            ensemble = w1 * probs_cnx + w2 * probs_eff
            preds = np.argmax(ensemble, axis=1)
            acc   = np.mean(preds == labels)

            if acc > best_acc:
                best_acc = acc
                best_w   = (round(w1, 3), round(w2, 3))

        print(f"[EnsembleOptimizer] Best weight → ConvNeXt: {best_w[0]:.3f} | EfficientNet: {best_w[1]:.3f}")
        print(f"[EnsembleOptimizer] Best accuracy pada val set: {best_acc * 100:.2f}%")

        # Update weight
        self.w_cnx = best_w[0]
        self.w_eff = best_w[1]

        return best_w


# ─────────────────────────────────────────────────────
# 5. Batch Inference (untuk generate submission Kaggle)
# ─────────────────────────────────────────────────────
def generate_submission(
    predictor: EnsemblePredictor,
    test_dir: str,
    output_csv: str = "submission.csv",
    use_tta: bool = True,
):
    """
    Generate file submission CSV untuk Kaggle.

    Args:
        predictor  : EnsemblePredictor yang sudah diload
        test_dir   : direktori berisi gambar test
        output_csv : path output CSV
        use_tta    : True = gunakan TTA (direkomendasikan untuk submission final)
    """
    import csv
    from pathlib import Path

    test_dir = Path(test_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([
        p for p in test_dir.rglob("*")
        if p.suffix.lower() in image_extensions
    ])

    print(f"[Submission] Memproses {len(image_paths)} gambar (TTA={'ON' if use_tta else 'OFF'})...")

    results = []
    for i, path in enumerate(image_paths):
        if i % 20 == 0:
            print(f"  {i}/{len(image_paths)} — {path.name}")
        try:
            label, probs = predictor.predict(str(path), use_tta=use_tta)
            confidence = float(np.max(probs))
            results.append({
                "filename": path.name,
                "label": label,
                "confidence": f"{confidence:.4f}",
            })
        except Exception as e:
            print(f"  [ERROR] {path.name}: {e}")
            results.append({
                "filename": path.name,
                "label": "realperson",  # default fallback
                "confidence": "0.1667",
            })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "confidence"])
        writer.writeheader()
        writer.writerows(results)

    print(f"[Submission] Saved ke {output_csv} ({len(results)} baris)")

    # Statistik distribusi prediksi
    from collections import Counter
    dist = Counter(r["label"] for r in results)
    print("[Submission] Distribusi prediksi:")
    for cls, count in sorted(dist.items()):
        print(f"  {cls:20s}: {count:4d} ({count/len(results)*100:.1f}%)")


# ─────────────────────────────────────────────────────
# 6. Contoh penggunaan
# ─────────────────────────────────────────────────────
USAGE_EXAMPLE = """
# ── Setup ──
from inference_v2 import EnsemblePredictor, generate_submission

predictor = EnsemblePredictor(
    convnext_path="checkpoints/convnext_best.pth",
    efficientnet_path="checkpoints/efficientnet_best.pth",
    convnext_weight=0.65,
    efficientnet_weight=0.35,
    temperature_convnext=1.3,
    temperature_efficientnet=1.2,
)

# Opsional: Optimasi weight pada validation set
# best_weights = predictor.optimize_weights(val_paths, val_labels)

# Generate submission
generate_submission(predictor, test_dir="data/test", use_tta=True)
"""

if __name__ == "__main__":
    print("inference_v2.py siap digunakan.")
    print(USAGE_EXAMPLE)
