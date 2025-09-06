# RFMiD Multi-Label Retinal Disease Benchmark  
**Backbones:** VGG16 · ResNet50 · EfficientNet-B0 · MobileNetV2

Multi-label classification of 45 retinal diseases on **RFMiD** under severe class imbalance.  
Protocol uses ImageNet initialization, positive-class **weighted BCE**, **per-class** F1-optimal threshold calibration (validation split), mixed precision, short warm-up (frozen) + fine-tuning last *N=4* layers.  
All plots are saved as **PDF/SVG** with enlarged fonts. Per-class thresholds and pretrained weights are exported per model.

---

## 1) TL;DR results (validation, per-class thresholds)

| Model             | Macro-F1 | Micro-F1 | Macro-AUC | Hamming Loss ↓ | EMR ↑  |
|-------------------|----------|----------|-----------|----------------|--------|
| VGG16             | 0.195    | 0.202    | 0.786     | 0.111          | 0.102  |
| ResNet50          | 0.196    | 0.258    | 0.781     | 0.083          | 0.122  |
| **EfficientNet-B0** | **0.222** | **0.346** | **0.799** | **0.055**        | **0.159** |
| MobileNetV2       | 0.221    | 0.267    | 0.747     | 0.079          | 0.075  |

Notes: Metrics computed on the **validation** split after per-class threshold calibration; the same frozen thresholds can be applied to the test set for final reporting.

---

## 2) Dataset layout (Kaggle)

/kaggle/input/retinal-disease-classification/
├── Training_Set/Training_Set/Training/
├── Evaluation_Set/Evaluation_Set/Validation/
├── Test_Set/Test_Set/Test/
├── Training_Set/Training_Set/RFMiD_Training_Labels.csv
├── Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv
└── Test_Set/Test_Set/RFMiD_Testing_Labels.csv


---

## 3) Environment

- Python ≥ 3.9, **TensorFlow ≥ 2.11**, Keras, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn  
- Mixed precision: `tf.keras.mixed_precision.set_global_policy("mixed_float16")`  
- GPU recommended (Kaggle: T4/P100)

---

## 4) Quick start (Kaggle)

1. Create a new **GPU** notebook and attach the RFMiD dataset above.  
2. Paste the training script/notebook cells (final code with the four backbones).  
3. Run all cells.

Key knobs:

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_WARMUP = 3         # backbone frozen
EPOCHS_FINETUNE = 7       # unfreeze last N=4 layers
LR = 1e-4                 # Adam
UNFREEZE_LAST_N = 4
THRESH_GRID = np.linspace(0.0, 1.0, 101)  # F1 sweep for τ_c
```
/kaggle/working/
├── figures/                         # Vector figures (PDF + SVG)
│   ├── fig_pos_counts.pdf|svg
│   ├── fig_model_compare_macroF1.pdf|svg
│   ├── fig_cm_flat_VGG16.pdf|svg
│   ├── fig_cm_flat_EfficientNetB0.pdf|svg
│   └── fig_pipeline.pdf|svg
├── logs/
│   ├── VGG16/
│   │   ├── weights.h5
│   │   ├── thresholds.csv           # per-class τ_c from validation
│   │   └── history.csv
│   ├── ResNet50/ …
│   ├── EfficientNetB0/ …
│   └── MobileNetV2/ …
├── reports/
│   ├── validation_metrics.csv       # macro/micro-F1, macro-AUC, HL, EMR
│   └── classification_report_*.txt
└── model_comparison.csv             # consolidated table across models

!zip -r /kaggle/working/benchmark_artifacts.zip \
      /kaggle/working/figures /kaggle/working/logs /kaggle/working/reports model_comparison.csv

FIG_DIR = Path("/kaggle/working/figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_DIR / "fig_model_compare_macroF1.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "fig_model_compare_macroF1.svg", bbox_inches="tight")

import numpy as np, pandas as pd, tensorflow as tf

model = tf.keras.models.load_model("logs/EfficientNetB0/weights.h5", compile=False)
taus = pd.read_csv("logs/EfficientNetB0/thresholds.csv", index_col=0)["tau"].values  # shape: (45,)

probs = model.predict(dataset, verbose=0)  # dataset: batched tf.data
pred  = (probs >= taus[None, :]).astype(int)



