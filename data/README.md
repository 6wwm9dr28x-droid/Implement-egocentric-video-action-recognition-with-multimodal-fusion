# Stage 1 — Data Preparation

**Team Member: Uzoma Eze**

## What was done

1. **Downloaded** EPIC-KITCHENS-100 videos and RGB frames for Participant P01 using the official Bristol download scripts
2. **Extracted** per-clip frame sequences aligned to `start_frame` / `stop_frame` timestamps
3. **Designed splits** — stratified by video identity so no video appears in two splits (prevents temporal leakage):
   - Train: 5,509 clips across 22 videos
   - Val: 885 clips across 5 videos
   - Test: 647 clips from P01_101 (no labels — prediction target)
   - Demo: 15 clips from P01_01 (with labels, for live demonstration)
4. **Analysed class imbalance** — verb distribution is severely long-tailed:
   - Most common: *put-down* (961), *take* (761), *pick-up* (523)
   - 59 total verb classes; many appear fewer than 10 times in train
5. **Prepared metadata CSVs** with columns:
   `narration_id, participant_id, video_id, start_frame, stop_frame, narration, verb, verb_class, noun, noun_class, action_label, duration_frames`

## Files

| File | Description |
|:---|:---|
| `train_metadata_prepared.csv` | 5,509 labeled training clips |
| `val_metadata_prepared.csv` | 885 labeled validation clips |
| `test_metadata_prepared.csv` | 647 unlabeled test clips (no narration / noun) |
| `demo_metadata_prepared.csv` | 15 demo clips from P01_01 |
| `dataset_summary.csv` | Split overview |
| `top20_verb.csv` | Top-20 verb frequencies |
| `top20_noun.csv` | Top-20 noun frequencies |
| `top20_action_label.csv` | Top-20 action label frequencies |

## Key finding

The test set contains **no narration and no noun labels** — only `video_id`, `start_frame`, and `stop_frame`. This means text and object features cannot be used at test time; only visual (CLIP) features are available for producing test predictions. This is by design — it prevents label leakage and forces a genuine visual recognition challenge.
