"""
Extract CLIP visual features locally from downloaded tars.
Only reads needed frames from each tar (no full extraction).
Produces: features/test_visual.npy, features/demo_visual.npy
"""
import os, tarfile, time
import numpy as np
import pandas as pd
import torch
from PIL import Image
import open_clip

DATA = "../team2_feature_extraction"
DOWNLOADS = "downloads"
OUT = "features"
os.makedirs(OUT, exist_ok=True)

TARS = {
    "P01_01":  f"{DOWNLOADS}/P01_01.tar",
    "P01_101": f"{DOWNLOADS}/P01_101.tar",
}
SPLITS_TO_RUN = ["demo", "test", "train", "val"]
NUMF = 3

device = "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
clip_model = clip_model.eval()
print("CLIP loaded on CPU")

@torch.no_grad()
def embed_images(imgs):
    if not imgs: return None
    x = torch.stack([preprocess(im) for im in imgs])
    f = clip_model.encode_image(x)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.numpy()

for split in SPLITS_TO_RUN:
    csv_path = f"{DATA}/{split}_metadata_prepared.csv"
    if not os.path.exists(csv_path): continue
    df = pd.read_csv(csv_path)
    if "narration" not in df.columns and split not in ["test"]:
        print(f"  {split}: skipping (no narration col)"); continue

    # find which videos in this split we have tars for
    available = [v for v in df["video_id"].unique() if v in TARS]
    if not available:
        print(f"  {split}: no downloaded videos, saving zeros")
        np.save(f"{OUT}/{split}_visual.npy", np.zeros((len(df), 512), dtype="float32"))
        continue

    VIS = np.zeros((len(df), 512), dtype="float32")
    for v in available:
        rows = df[df["video_id"] == v]
        needed = set()
        clip_map = {}
        for i, row in rows.iterrows():
            nums = list(np.linspace(int(row["start_frame"]), int(row["stop_frame"]), NUMF).astype(int))
            clip_map[i] = nums
            needed.update(nums)

        print(f"  {split}/{v}: scanning tar for {len(needed)} frames from {len(rows)} clips...", end=" ", flush=True)
        t0 = time.time()
        imgs = {}
        with tarfile.open(TARS[v]) as tar:
            for m in tar:
                bn = os.path.basename(m.name)
                if not bn.startswith("frame_"): continue
                try: fn = int(bn.split("_")[1].split(".")[0])
                except: continue
                if fn in needed:
                    imgs[fn] = Image.open(tar.extractfile(m)).convert("RGB").copy()
        print(f"loaded {len(imgs)} frames in {time.time()-t0:.0f}s, encoding...", end=" ", flush=True)

        keys = list(imgs.keys())
        E = embed_images([imgs[k] for k in keys])
        emap = {fn: E[j] for j, fn in enumerate(keys)} if E is not None else {}

        for i, nums in clip_map.items():
            vecs = [emap[n] for n in nums if n in emap]
            VIS[i] = np.mean(vecs, axis=0) if vecs else np.zeros(512, dtype="float32")
        print(f"done.")

    np.save(f"{OUT}/{split}_visual.npy", VIS)
    print(f"  -> saved {split}_visual.npy {VIS.shape}")

print("\nAll visual features saved.")
