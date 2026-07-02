"""
Team 3 - Phase 1 feature generation (text + object) for train/val.

NOTE: test set has no narration / noun_class, so text+object features are
only generated for train and val. Test predictions require the visual branch.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

DATA = "../team2_feature_extraction"
OUT = "features"
os.makedirs(OUT, exist_ok=True)

train = pd.read_csv(f"{DATA}/train_metadata.csv")
val = pd.read_csv(f"{DATA}/val_metadata.csv")
print(f"train {train.shape} | val {val.shape}")

# ---------- 1. Text embeddings (Sentence-BERT, 384-d) ----------
from sentence_transformers import SentenceTransformer
text_model = SentenceTransformer("all-MiniLM-L6-v2")

def text_emb(df, split):
    texts = df["narration"].fillna("").astype(str).tolist()
    emb = text_model.encode(texts, batch_size=64, convert_to_numpy=True,
                            show_progress_bar=False)
    np.save(f"{OUT}/{split}_text.npy", emb)
    print(f"  {split} text {emb.shape}")
    return emb

print("Text embeddings...")
text_emb(train, "train")
text_emb(val, "val")

# ---------- 2. Object features (one-hot noun_class) ----------
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
tr_obj = ohe.fit_transform(train[["noun_class"]].astype(str))
va_obj = ohe.transform(val[["noun_class"]].astype(str))
np.save(f"{OUT}/train_object.npy", tr_obj)
np.save(f"{OUT}/val_object.npy", va_obj)
print(f"Object features: train {tr_obj.shape} | val {va_obj.shape}")

# ---------- 3. Labels (verb_class, noun_class), fit on train ----------
def encode_labels(col):
    le = LabelEncoder()
    tr = le.fit_transform(train[col].astype(str))
    known = set(le.classes_)
    va = np.array([le.transform([x])[0] if x in known else -1
                   for x in val[col].astype(str)])
    np.save(f"{OUT}/train_{col}.npy", tr)
    np.save(f"{OUT}/val_{col}.npy", va)
    pd.DataFrame({"class_name": le.classes_,
                  "class_id": range(len(le.classes_))}
                 ).to_csv(f"{OUT}/{col}_mapping.csv", index=False)
    n_unseen = int((va == -1).sum())
    print(f"  {col}: {len(le.classes_)} classes | val unseen rows: {n_unseen}")

print("Labels...")
for c in ["verb_class", "noun_class"]:
    encode_labels(c)

print("DONE. Features in ./features/")
