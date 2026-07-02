"""
Team 3 - Phase 2: train video-only model + fusion, produce test predictions.
Note: only P01_01 frames available for train (329/5509 clips have visual).
Val visual = zeros (no frames downloaded). Test visual = full 647 clips.
"""
import json, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import f1_score
torch.manual_seed(0); np.random.seed(0)

F = "features"
DATA = "../team2_feature_extraction"
def L(s, k): return np.load(f"{F}/{s}_{k}.npy")

train = pd.read_csv(f"{DATA}/train_metadata.csv")
val   = pd.read_csv(f"{DATA}/val_metadata.csv")
test  = pd.read_csv(f"{DATA}/test_metadata.csv")

N_VERB = int(L("train","verb_class").max()) + 1
N_NOUN = int(L("train","noun_class").max()) + 1

# mask: only train rows that have real visual (non-zero)
vis_tr = L("train","visual")
has_vis = np.any(vis_tr != 0, axis=1)
print(f"Train rows with real visual: {has_vis.sum()} / {len(train)}")

def feat(split, mode, mask=None):
    parts = []
    if "t" in mode: parts.append(L(split,"text") if split != "test" else np.zeros((len(test),384),dtype="float32"))
    if "o" in mode: parts.append(L(split,"object") if split != "test" else np.zeros((len(test),123),dtype="float32"))
    if "v" in mode: parts.append(L(split,"visual"))
    X = np.concatenate(parts, axis=1).astype("float32")
    return X[mask] if mask is not None else X

def cw(y, n):
    c = np.bincount(y, minlength=n).astype(float)
    w = 1.0 / np.clip(c, 1, None)
    return torch.tensor(w / w.sum() * n, dtype=torch.float32)

class Net(nn.Module):
    def __init__(self, d, nv, nn_):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.3))
        self.verb = nn.Linear(128, nv)
        self.noun = nn.Linear(128, nn_)
    def forward(self, x):
        h = self.trunk(x); return self.verb(h), self.noun(h)

def topk(logits, y, k, m):
    if m.sum() == 0: return float("nan")
    tk = logits.topk(k,1).indices.numpy()[m]; yy = y[m]
    return float(np.mean([yy[i] in tk[i] for i in range(len(yy))]))

def run(mode, epochs=60, use_mask=False):
    mask = has_vis if use_mask else None
    Xtr = torch.tensor(feat("train", mode, mask))
    vtr = torch.tensor(L("train","verb_class")[mask] if mask is not None else L("train","verb_class"))
    ntr = torch.tensor(L("train","noun_class")[mask] if mask is not None else L("train","noun_class"))
    Xva = torch.tensor(feat("val", mode))
    wv = cw(vtr.numpy(), N_VERB); wn = cw(ntr.numpy(), N_NOUN)
    net = Net(Xtr.shape[1], N_VERB, N_NOUN)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    lv = nn.CrossEntropyLoss(weight=wv); ln = nn.CrossEntropyLoss(weight=wn)
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]; opt.zero_grad()
            pv, pn = net(Xtr[idx])
            (lv(pv, vtr[idx]) + ln(pn, ntr[idx])).backward(); opt.step()
    net.eval()
    with torch.no_grad(): pv, pn = net(Xva)
    va_v = L("val","verb_class"); va_n = L("val","noun_class")
    mv = va_v >= 0; mn = va_n >= 0; m = mv & mn
    pvn = pv.argmax(1).numpy(); pnn = pn.argmax(1).numpy()
    r = {"mode": mode,
         "verb@1":  round(topk(pv, va_v, 1, mv), 3),
         "verb@5":  round(topk(pv, va_v, 5, mv), 3),
         "verbF1":  round(f1_score(va_v[mv], pvn[mv], average="macro", zero_division=0), 3),
         "noun@1":  round(topk(pn, va_n, 1, mn), 3),
         "noun@5":  round(topk(pn, va_n, 5, mn), 3),
         "nounF1":  round(f1_score(va_n[mn], pnn[mn], average="macro", zero_division=0), 3),
         "action@1":round(float(np.mean((pvn[m]==va_v[m])&(pnn[m]==va_n[m]))), 3)}
    return net, r

print("Training ablation models...")
results = {}
# text + object (already done but re-run for complete table)
_, results["text"]   = run("t")
_, results["object"] = run("o")
_, results["to"]     = run("to")
# video-only (trained on 329 P01_01 clips only)
net_v, results["video"] = run("v", use_mask=True)
print("Done. Results:")
print(f"{'mode':8} {'verb@1':>7} {'verb@5':>7} {'verbF1':>7} {'noun@1':>7} {'noun@5':>7} {'nounF1':>7} {'act@1':>7}")
for k, r in results.items():
    print(f"{r['mode']:8} {r['verb@1']:7.3f} {r['verb@5']:7.3f} {r['verbF1']:7.3f} "
          f"{r['noun@1']:7.3f} {r['noun@5']:7.3f} {r['nounF1']:7.3f} {r['action@1']:7.3f}")

json.dump(list(results.values()), open("metrics_final.json","w"), indent=2)

# Test predictions using video-only model (only valid input for test set)
print("\nGenerating test predictions...")
Xte = torch.tensor(feat("test","v"))
net_v.eval()
with torch.no_grad(): pv, pn = net_v(Xte)
pv = pv.argmax(1).numpy(); pn = pn.argmax(1).numpy()

verb_le = pd.read_csv(f"{F}/verb_class_mapping.csv")
noun_le = pd.read_csv(f"{F}/noun_class_mapping.csv")
vmap = dict(zip(verb_le.class_id, verb_le.class_name))
nmap = dict(zip(noun_le.class_id, noun_le.class_name))

out = test[["narration_id","video_id","start_frame","stop_frame"]].copy()
out["pred_verb_class"] = pv
out["pred_noun_class"] = pn
out["pred_verb"] = [vmap.get(x,"unknown") for x in pv]
out["pred_noun"] = [nmap.get(x,"unknown") for x in pn]
out["pred_action"] = out["pred_verb"].astype(str) + " " + out["pred_noun"].astype(str)
out.to_csv("test_predictions.csv", index=False)
print(f"Saved test_predictions.csv ({len(out)} rows)")
print(out[["narration_id","pred_verb","pred_noun","pred_action"]].head(10).to_string())
