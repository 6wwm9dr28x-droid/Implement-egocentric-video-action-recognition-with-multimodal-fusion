# Stage 3 — Model Development & Evaluation

**Team Member: Magomed Makhsudov**

## Architecture

```
 narration ──► [Sentence-BERT, frozen] ──► 384-d ──┐
                                                     │ concat
 noun_class ──► [one-hot, 123 classes] ──► 123-d ──┤
                                                     │
 frames ──────► [CLIP ViT-B/32, frozen] ─► 512-d ──┘ (optional)
                              │
            ┌─────────────────▼──────────────────────┐
            │  Linear(d_in → 256) · BN · ReLU · Drop  │
            │  Linear(256  → 128)       · ReLU · Drop  │
            └──────────────┬──────────────┬───────────┘
                           ▼              ▼
                    verb head (59)  noun head (123)
                    weighted CE     weighted CE
```

**Two-head design** — predicting verb and noun separately (59 + 123 classes) generalises far better than the flat 1,060 action labels (many unseen in val). Action prediction = verb AND noun both correct.

**Weighted cross-entropy** — inverse-frequency class weights; without this the model collapses to predicting the most common verb (*put-down*) for every clip.

## Training

| Hyperparameter | Value |
|:---|:---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Epochs | 40–60 |
| Batch size | 256 |
| Loss | `weighted_CE(verb) + weighted_CE(noun)` |

## Ablation

Four configurations trained and compared:

| Config | Input dim | Purpose |
|:---|:---:|:---|
| Text only | 384 | Upper-bound baseline (leaky) |
| Object only | 123 | Structural prior baseline |
| **Fusion (text + object)** | 507 | Primary model |
| Video only | 512 | Honest deployable baseline |

## Scripts

| Script | What it does |
|:---|:---|
| `generate_features.py` | Generate text + object features + labels for train/val/demo |
| `extract_visual_features.py` | Stream CLIP visual features from downloaded frame tars |
| `train_fusion.py` | Train all ablation models, save `metrics_all_models.json` |
| `train_video.py` | Train video-only model, produce `test_predictions.csv` |
| `visualize_results.py` | Generate presentation charts |
