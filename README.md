# Multimodal Egocentric Action Recognition on EPIC-KITCHENS

**Project 11 · Machine Learning · Group 11**

Can we figure out what someone is doing in the kitchen just by watching from their point of view? That's the question we set out to answer. Using short first-person clips from the EPIC-KITCHENS-100 dataset, we built a model that combines what people *say* they're doing, what objects are in the scene, and what the camera actually sees — then predicts the action.

## The Team

We split the work three ways, playing to what each of us wanted to dig into:

- **Uzoma Eze** handled the data side — downloading the dataset, pulling out frames, designing the train/val splits, and digging into the class imbalance mess.
- **Amrin Yanya** worked on feature extraction — wiring up Sentence-BERT for the narrations, CLIP for the visual side, and building the one-hot object features.
- **Magomed Makhsudov** built the actual model — the fusion architecture, training pipeline, ablations, and the final test predictions.

## What We're Actually Doing

We scoped the problem to a single participant (P01) from EPIC-KITCHENS-100. Every clip is a few seconds of someone cooking, filmed from a head-mounted camera. The model has to predict a **verb** (like *cut*) and a **noun** (like *tomato*). Together they make the action (*cut tomato*).

A few decisions shaped everything else:

- **We didn't fine-tune the big encoders.** Sentence-BERT and CLIP stay frozen. We only train the small fusion head on top. Less compute, less overfitting risk, and it turned out to be plenty.
- **Two heads instead of one.** Predicting verb (59 classes) and noun (123 classes) separately is way easier than predicting one of 1,060 flat action labels. And you can still combine them at the end.
- **The long tail is brutal.** Some actions show up hundreds of times; others show up twice. We used inverse-frequency weighted loss to stop the model from just memorizing the common classes.
- **Accuracy alone lies.** With this much imbalance, top-1 accuracy makes bad models look good. We report macro-F1 too, which actually cares about the rare classes.

## The Data

| Split | Clips | Videos | Labels |
|-------|------:|-------:|--------|
| Train | 5,509 | 22 | verb + noun |
| Val   |   885 | 5  | verb + noun |
| Test  |   647 | 1 (P01_101) | — (what we predict) |
| Demo  |    15 | 1 (P01_01)  | verb + noun |

Everything comes from the [Bristol Research Data Repository](https://data.bris.ac.uk/data/dataset/2g1n6qdydwa9u22shpxqzp0t8m).

## How the Pipeline Fits Together

```
Raw Videos (EPIC-KITCHENS P01)
        │
        ▼
┌───────────────────────────────────┐
│  Stage 1 — Data Prep              │  Uzoma
│  Frame extraction, splits,        │
│  class imbalance analysis         │
└────────────────┬──────────────────┘
                 ▼
┌───────────────────────────────────┐
│  Stage 2 — Feature Extraction     │  Amrin
│                                   │
│  narration ─► Sentence-BERT ─► 384-d
│  noun_class ─► one-hot ──────► 123-d
│  frames ────► CLIP ViT-B/32 ─► 512-d
└────────────────┬──────────────────┘
                 ▼
┌───────────────────────────────────┐
│  Stage 3 — Fusion Model           │  Magomed
│                                   │
│  concat(text, object [, visual])  │
│    → MLP (256 → 128)              │
│    → verb head (59)               │
│    → noun head (123)              │
└────────────────┬──────────────────┘
                 ▼
        Results & Predictions
```

## What We Found

All numbers on the validation set (885 clips):

| Model | Verb@1 | Verb@5 | Verb F1 | Noun@1 | Noun@5 | Noun F1 | Action@1 |
|-------|-------:|-------:|--------:|-------:|-------:|--------:|---------:|
| Text only              | 0.973 | 0.998 | 0.787 | 0.967 | 0.990 | 0.831 | 0.944 |
| Object only            | 0.122 | 0.704 | 0.085 | 1.000 | 1.000 | 1.000 | 0.129 |
| **Fusion (text + object)** | **0.982** | **1.000** | **0.918** | **1.000** | **1.000** | **1.000** | **0.986** |
| Video only†            | 0.059 | 0.595 | 0.003 | 0.042 | 0.053 | 0.001 | 0.000 |

† Video-only was trained on just 329 clips from P01_01 — we ran into download limits and couldn't get the rest of the frames in time.

**The headline result:** fusing text with object features beats every single-stream model on macro-F1 by a wide margin (+16.6 pp over text alone). The object features rescue exactly the tail classes where the narration is ambiguous — which is what you'd hope for.

**An honest caveat:** the text narrations are basically leaking the label. They're descriptive by design, so a text-only model does suspiciously well. In a real deployment you wouldn't have those narrations, which is why the video-only baseline — bad as it looks here — is really the one that matters. The fusion result is best read as an upper bound on what multimodal signals can offer, not the final production model.

## What's in the Repo

```
├── data/         Stage 1 — metadata CSVs, class distribution analysis
├── features/     Stage 2 — feature extraction code + notebooks
├── model/        Stage 3 — training scripts, fusion architecture
├── results/      Metrics, test predictions, plots for the presentation
├── notebooks/    End-to-end runnable presentation notebook
└── README.md
```

## Running It Yourself

Install the dependencies:

```bash
pip install numpy pandas scikit-learn sentence-transformers torch open_clip_torch matplotlib pillow
```

Then run the three stages:

```bash
# 1. Extract text + object features for train / val / demo
python model/generate_features.py

# 2. Train all the models and produce the ablation table
python model/train_fusion.py

# 3. Make the plots we used in the presentation
python model/visualize_results.py
```

Extracting the CLIP visual features is more involved — you need to pull the P01 frame tarballs from the Bristol server first. See [`features/README.md`](features/README.md) for the full workflow.

## The Presentation Notebook

If you just want to see the whole thing end-to-end, open [`notebooks/multimodal_action_recognition.ipynb`](notebooks/multimodal_action_recognition.ipynb). It's self-contained, runs on CPU in about 30 seconds, and walks through:

- loading the pre-extracted features from `results/features/`,
- training the fusion model live,
- evaluating on the validation set,
- printing the ablation table and plots,
- and showing predictions on the demo clip.

---

*Group 11 · Machine Learning · 2026*
