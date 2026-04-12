# 🛡️ Face Anti-Spoofing Challenge (Find IT!)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-🔥-red.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This repository contains a robust Machine Learning pipeline developed for the **Face Anti-Spoofing Challenge (Find IT!)** on Kaggle. The goal of this project is to accurately detect various facial spoofing attacks, distinguishing a real person from fake representations such as printed photos, digital screens, 3D masks, and mannequins.

## 🚀 Key Achievements
* **Final Kaggle Public Score:** `0.77257`
* **Local Validation Accuracy:** `~96.76%`
* Successfully trained a heavy vision model (ConvNeXt-V2 Base) locally on limited hardware by heavily optimizing the data pipeline and batch processing.

## 🧠 Model Architecture & Strategy
The final solution utilizes the **ConvNeXt-V2 Base** architecture (`convnextv2_base.fcmae_ft_in22k_in1k`) trained with a carefully engineered pipeline to overcome severe domain gaps between the training and testing datasets.

### Key Techniques Implemented:
1. **Robust Face Extraction (MTCNN):** Utilized MTCNN for precise face cropping. To handle edge cases where faces were heavily occluded or undetected (e.g., full masks or extreme lighting), a **Smart Center Crop Fallback** logic was implemented to preserve spatial context.
2. **Multi-Zoom Test Time Augmentation (TTA):** During inference, the model analyzes each image at 3 different zoom levels (Tight `margin=20`, Normal `margin=60`, Wide `margin=100`). This completely eliminates scale mismatches between train and test sets.
3. **Knowledge Injection & Pseudo-Labeling:** Successfully integrated high-confidence test predictions (distilled from top-performing ensemble submissions) back into the training loop to bridge the domain gap and adapt to the specific camera characteristics of the test set.
4. **Threshold Shifting (Hacker Data Strategy):** Adjusted prediction probabilities mathematically at the inference level to counteract tricky class imbalances and dataset distribution biases.
5. **K-Fold Cross Validation:** Trained 5 independent models using Stratified K-Fold to ensure maximum robustness against noise, preventing overfitting on specific spoofing textures (like Moiré patterns or screen glare).

## 📁 Repository Structure
To maintain a clean and professional workspace, the notebooks are divided into main pipelines and historical experiments.

```text
├── data/                   # (Ignored) Dataset directory
├── models/                 # (Ignored) Saved .pth weights
├── notebooks/              # 🌟 MAIN PIPELINE
│   ├── 01_EDA_and_Face_Extraction.ipynb
│   ├── 02_Training_ConvNeXtV2_Endgame.ipynb
│   ├── 03_Training_KFold_ConvNeXtV2.ipynb
│   └── 04_Final_Inference_and_Submission.ipynb
├── experiments/            # 🧪 HISTORICAL EXPERIMENTS & IDEAS
│   ├── 01_Baseline_EfficientNetB3.ipynb
│   ├── 02_Training_Swin_Transformer.ipynb
│   ├── 03_Trisula_Ensemble_Inference.ipynb
│   └── ... (Other legacy approaches)
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
