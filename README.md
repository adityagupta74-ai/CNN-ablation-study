# Progressive CNN Classifier — CIFAR-10

A step-by-step study of deep learning regularization and optimization techniques, built on CIFAR-10. Instead of training one "final" model, this project starts from a plain CNN and adds **one technique at a time** — measuring the actual, isolated impact of each on accuracy, loss, and overfitting.

## Project Structure

Each stage keeps the previous architecture and training conditions identical, changing only the one variable under test, so results are directly comparable.

| # | Notebook | Technique Added | Test Accuracy | Test Loss |
|---|---|---|---|---|
| 01 | `01_baseline_cnn.ipynb` | None (plain CNN) | 0.7166 | 0.8821 |
| 02 | `02_batchnorm.ipynb` | + Batch Normalization | 0.6807 | 1.5024 |
| 03 | `03_dropout.ipynb` | + Dropout | 0.7250 | 0.7805 |
| 04 | `04_data_augmentation.ipynb` | + Data Augmentation (15 epochs) | 0.6359 | 1.0204 |
| 04b | `04b_data_augmentation_longrun.ipynb` | + Data Augmentation (35 epochs) | 0.6680 | 0.9441 |
| 05 | `05_lr_scheduling.ipynb` | + LR Scheduling (ReduceLROnPlateau) | **0.7407** | **0.7296** |
| 06 | `06_transfer_learning.ipynb` | Transfer Learning (MobileNetV2) | **0.8928** | **0.3232** |

## Key Findings

- **BatchNorm alone made things worse.** It sped up training so aggressively that the model overfit faster than the plain baseline (train accuracy hit 93% while validation plateaued around 65%).
- **Dropout fixed it.** Combined with BatchNorm, Dropout eliminated the overfitting entirely and pushed accuracy *above* the original baseline, with train/val curves tracking closely throughout training.
- **Data Augmentation underperformed at 15 epochs**, but not from overfitting — the model was still improving with no train/val divergence when training stopped. A 35-epoch follow-up run recovered some of that gap but plateaued below the Dropout model, suggesting CIFAR-10's low 32×32 resolution limits how much aggressive geometric augmentation actually helps.
- **LR Scheduling gave the best from-scratch result (74.07%)** — `ReduceLROnPlateau` let the optimizer recover from a late-training loss spike and settle into a sharper minimum.
- **Transfer Learning dominated everything.** Fine-tuning a MobileNetV2 backbone pretrained on ImageNet reached **89.28% accuracy**, a 15+ point jump over the best model trained from scratch — a clear demonstration of how much pretrained features can accelerate results on a smaller dataset.

## Setup

\`\`\`bash
git clone https://github.com/adityagupta74-ai/progressive-cnn-classifier.git
cd progressive-cnn-classifier
python3 -m venv DLenv
source DLenv/bin/activate
pip install tensorflow numpy matplotlib scikit-learn seaborn jupyter ipykernel
\`\`\`

Each notebook is self-contained — open in Jupyter or VS Code and run top to bottom. CIFAR-10 downloads automatically on first run (~170MB, cached afterward).

## Tech Stack

- Python, TensorFlow / Keras
- CIFAR-10 dataset (60,000 32×32 color images, 10 classes)
- Trained on a MacBook Air (CPU)

## Possible Extensions

- Combine LR scheduling with transfer learning fine-tuning
- Ensemble the top 3 models (03, 05, 06)
- Add Grad-CAM visualizations for interpretability
- Compare against a second pretrained backbone (e.g., EfficientNet)
