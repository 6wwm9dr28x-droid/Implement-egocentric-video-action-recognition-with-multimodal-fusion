# Stage 2 — Feature Extraction

**Team Member: Amrin Yanya**

## Design principle

All encoders are **frozen** — no fine-tuning. This keeps the pipeline feasible within compute constraints while leveraging powerful general-purpose representations.

## Three feature streams

### Text (384-d) — `*_text.npy`
- **Encoder:** Sentence-BERT `all-MiniLM-L6-v2`
- **Input:** Free-text `narration` field (e.g. *"open door"*)
- **Output:** 384-dimensional dense embedding per clip
- **Available for:** train, val, demo — **not test** (test has no narration)

### Object (123-d) — `*_object.npy`
- **Encoder:** One-hot encoding of `noun_class` (123 categories)
- **Input:** Integer noun class label
- **Output:** 123-dimensional sparse binary vector per clip
- **Available for:** train, val, demo — **not test** (test has no noun label)
- **Note:** This is a lightweight object cue; a full object detector would replace this in production

### Visual (512-d) — `*_visual.npy`
- **Encoder:** CLIP `ViT-B/32` (OpenAI pretrained), frozen
- **Input:** 3 frames uniformly sampled from `[start_frame, stop_frame]`
- **Output:** Per-frame 512-d features → L2-normalised → mean-pooled → 512-d per clip
- **Available for:** all splits (only requires frame tars from Bristol server)
- **Frame path convention:** `frames/{video_id}/frame_{n:010d}.jpg`

## Output files

```
features/
├── train_text.npy          (5509, 384)
├── train_object.npy        (5509, 123)
├── train_visual.npy        (5509, 512)  ← requires frame download
├── train_verb_class.npy    (5509,)
├── train_noun_class.npy    (5509,)
├── val_text.npy            (885, 384)
├── val_object.npy          (885, 123)
├── val_visual.npy          (885, 512)   ← requires frame download
├── val_verb_class.npy      (885,)
├── val_noun_class.npy      (885,)
├── test_visual.npy         (647, 512)   ← requires P01_101 frames
├── demo_text.npy           (15, 384)
├── demo_object.npy         (15, 123)
├── demo_visual.npy         (15, 512)    ← requires P01_01 frames
├── verb_class_mapping.csv
└── noun_class_mapping.csv
```

## Running extraction

```bash
# Text + object (no download needed)
python model/generate_features.py

# Visual (requires frame tars from Bristol server)
# Download P01_01 (~1.7 GB) and P01_101 (~1.7 GB), then:
python model/extract_visual_features.py
```

See `model/extract_visual_features.py` for streaming single-pass extraction (reads only needed frames from each tar without full unpack).
