# Multimodal Egocentric Action Recognition — EPIC-KITCHENS

**Project 11 | Machine Learning Course | Group 11**

> *Can we recognise fine-grained cooking actions from egocentric video by fusing language narrations, object cues, and video frames using frozen pretrained encoders?*

---

## Team

| Member | Role | Contribution |
|:---|:---|:---|
| Uzoma Eze | Data Preparation | Dataset download, frame extraction, split design, class-imbalance analysis |
| Amrin Yanya | Feature Extraction | Sentence-BERT text encoder, CLIP visual encoder, one-hot object features |
| Magomed Makhsudov | Model Development & Evaluation | Fusion architecture, training, ablation study, test predictions |

---

## Overview

We tackle **action recognition** on the [EPIC-KITCHENS-100](https://epic-kitchens.github.io) dataset, restricted to Participant P01. Given a short first-person video clip, the model predicts the **verb** (e.g. *cut*) and **noun** (e.g. *tomato*) that together form the action (e.g. *cut tomato*).

**Key design choices:**
- Freeze large pretrained encoders (Sentence-BERT, CLIP) — no fine-tuning
- Predict verb and noun with **two separate heads** (59 + 123 classes) rather than the flat 1,060 action labels
- Handle severe long-tail imbalance with **inverse-frequency weighted loss**
- Evaluate with **macro-F1** alongside top-1/5 accuracy to capture tail-class performance

---

## Dataset

| Split | Clips | Videos | Labels |
|:---|---:|---:|:---|
| Train | 5,509 | 22 | verb + noun |
| Val | 885 | 5 | verb + noun |
| Test | 647 | 1 (P01_101) | none — prediction target |
| Demo | 15 | 1 (P01_01) | verb + noun |

Source: [Bristol Research Data Repository](https://data.bris.ac.uk/datasets/)

---

## Pipeline

```
Raw Videos (EPIC-KITCHENS P01)
        │
        ▼
┌───────────────────────────────────┐
│  Stage 1 — Data Preparation       │  Uzoma Eze
│  Frame extraction · Split design  │
│  Class imbalance analysis         │
└────────────────┬──────────────────┘
                 │
                 ▼
┌───────────────────────────────────┐
│  Stage 2 — Feature Extraction     │  Amrin Yanya
│                                   │
│  narration ─► Sentence-BERT ──► 384-d text vector
│  noun_class ─► one-hot ───────► 123-d object vector
│  frames ─────► CLIP ViT-B/32 ─► 512-d visual vector
└────────────────┬──────────────────┘
                 │
                 ▼
┌───────────────────────────────────┐
│  Stage 3 — Fusion Model           │  Magomed Makhsudov
│                                   │
│  concat(text, object [, visual])  │
│       → MLP trunk (256 → 128)     │    
│       → verb head (59 classes)    │
│       → noun head (123 classes)   │
└────────────────┬──────────────────┘
                 │
                 ▼
        Results & Predictions
```

---

## Results

All metrics on the **validation set** (885 clips):

| Model | Verb @1 | Verb @5 | Verb F1 | Noun @1 | Noun @5 | Noun F1 | Action @1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Text only | 0.973 | 0.998 | 0.787 | 0.967 | 0.990 | 0.831 | 0.944 |
| Object only | 0.122 | 0.704 | 0.085 | 1.000 | 1.000 | 1.000 | 0.129 |
| **Fusion (text + object)** | **0.982** | **1.000** | **0.918** | **1.000** | **1.000** | **1.000** | **0.986** |
| Video only† | 0.059 | 0.595 | 0.003 | 0.042 | 0.053 | 0.001 | 0.000 |

† Trained on 329 clips from P01_01 only due to download constraints.

**Key finding:** Fusion beats every single stream on macro-F1 (+16.6 pp over text-only), confirming that object cues rescue tail-class predictions where narrations alone are ambiguous. Text narrations are near-label-leaky by design — the video-only baseline is the honest deployable model.

---

## Repository Structure

```
├── data/               Stage 1 — metadata CSVs, class distribution analysis
├── features/           Stage 2 — feature extraction code and notebook
├── model/              Stage 3 — training scripts, fusion architecture
├── results/            Metrics, test predictions, presentation plots
├── notebooks/          Full presentation notebook (end-to-end runnable)
└── README.md
```

---

## Running the Code

### Requirements
```bash
pip install numpy pandas scikit-learn sentence-transformers torch open_clip_torch matplotlib pillow
```

### Step-by-step

```bash
# 1. Generate text + object features (train / val / demo)
python model/generate_features.py

# 2. Train all models + ablation table
python model/train_fusion.py

# 3. Generate presentation plots
python model/visualize_results.py
```

> Visual (CLIP) feature extraction requires downloading P01 frame tars from the Bristol server.
> See `features/README.md` for the full extraction workflow.

---

## Presentation Notebook

`notebooks/multimodal_action_recognition.ipynb` — fully self-contained, runs on CPU in ~30 s:

- Loads pre-extracted features from `results/features/`
- Defines and trains the fusion model live
- Evaluates on val, shows ablation table, plots, and demo clip predictions

---

*Group 11 | Machine Learning | 2026*
