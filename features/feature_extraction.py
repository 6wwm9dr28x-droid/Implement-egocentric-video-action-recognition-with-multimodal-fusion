"""
Team Member 2 - Feature Extraction Pipeline
Project: Video-and-Language Action Recognition in Egocentric Kitchens

This script prepares text, visual, object, and label features for Team Member 3.
Run in Colab/Kaggle after installing:
    pip install sentence-transformers open_clip_torch opencv-python torch torchvision pillow scikit-learn
"""

import os
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

TRAIN_CSV = "train_metadata.csv"
VAL_CSV = "val_metadata.csv"
TEST_CSV = "test_metadata.csv"
OUTPUT_DIR = "features"
FRAMES_ROOT = "frames"  # expected structure: frames/{video_id}/frame_0000000001.jpg

os.makedirs(OUTPUT_DIR, exist_ok=True)


def prepare_metadata(csv_path: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "verb" in df.columns and "noun" in df.columns:
        df["action_label"] = df["verb"].astype(str) + " " + df["noun"].astype(str)
        df["duration_frames"] = df["stop_frame"] - df["start_frame"] + 1
    df.to_csv(os.path.join(OUTPUT_DIR, f"{split}_metadata_prepared.csv"), index=False)
    return df


def extract_text_embeddings(df: pd.DataFrame, split: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = df["narration"].fillna("").astype(str).tolist()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    np.save(os.path.join(OUTPUT_DIR, f"{split}_text_embeddings.npy"), embeddings)
    return embeddings


def sample_frame_paths(row, num_frames: int = 3):
    video_id = row["video_id"]
    start = int(row["start_frame"])
    stop = int(row["stop_frame"])
    frame_numbers = np.linspace(start, stop, num_frames).astype(int)
    paths = []
    for fno in frame_numbers:
        # Change this pattern if Team Member 1 used a different frame filename format.
        candidate = os.path.join(FRAMES_ROOT, video_id, f"frame_{fno:010d}.jpg")
        paths.append(candidate)
    return paths


def extract_visual_embeddings(df: pd.DataFrame, split: str, num_frames: int = 3):
    import torch
    from PIL import Image
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device)
    model.eval()

    all_features = []
    missing = 0

    with torch.no_grad():
        for _, row in df.iterrows():
            frame_paths = sample_frame_paths(row, num_frames=num_frames)
            frame_features = []
            for path in frame_paths:
                if not os.path.exists(path):
                    missing += 1
                    continue
                image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                feat = model.encode_image(image)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                frame_features.append(feat.cpu().numpy()[0])

            if frame_features:
                clip_feature = np.mean(frame_features, axis=0)
            else:
                clip_feature = np.zeros(512, dtype=np.float32)
            all_features.append(clip_feature)

    visual_embeddings = np.vstack(all_features)
    np.save(os.path.join(OUTPUT_DIR, f"{split}_visual_embeddings.npy"), visual_embeddings)
    print(f"Missing frame files for {split}: {missing}")
    return visual_embeddings


def extract_object_features(train_df: pd.DataFrame, val_df: pd.DataFrame):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_objects = train_df[["noun_class"]].astype(str)
    val_objects = val_df[["noun_class"]].astype(str)
    train_obj = encoder.fit_transform(train_objects)
    val_obj = encoder.transform(val_objects)
    np.save(os.path.join(OUTPUT_DIR, "train_object_features.npy"), train_obj)
    np.save(os.path.join(OUTPUT_DIR, "val_object_features.npy"), val_obj)
    return train_obj, val_obj


def save_labels(train_df: pd.DataFrame, val_df: pd.DataFrame):
    for label_col in ["verb_class", "noun_class", "action_label"]:
        le = LabelEncoder()
        train_labels = le.fit_transform(train_df[label_col].astype(str))
        val_labels = le.transform(val_df[label_col].astype(str)) if label_col != "action_label" else np.array([
            le.transform([x])[0] if x in le.classes_ else -1 for x in val_df[label_col].astype(str)
        ])
        np.save(os.path.join(OUTPUT_DIR, f"train_{label_col}_labels.npy"), train_labels)
        np.save(os.path.join(OUTPUT_DIR, f"val_{label_col}_labels.npy"), val_labels)
        pd.DataFrame({"class_name": le.classes_, "class_id": range(len(le.classes_))}).to_csv(
            os.path.join(OUTPUT_DIR, f"{label_col}_label_mapping.csv"), index=False
        )


def main():
    train_df = prepare_metadata(TRAIN_CSV, "train")
    val_df = prepare_metadata(VAL_CSV, "val")

    print("Extracting text embeddings...")
    extract_text_embeddings(train_df, "train")
    extract_text_embeddings(val_df, "val")

    print("Extracting object features...")
    extract_object_features(train_df, val_df)

    print("Saving labels...")
    save_labels(train_df, val_df)

    print("Visual feature extraction is ready. Run it after frames are available in FRAMES_ROOT.")
    # Uncomment when frames are available:
    # extract_visual_embeddings(train_df, "train", num_frames=3)
    # extract_visual_embeddings(val_df, "val", num_frames=3)


if __name__ == "__main__":
    main()
