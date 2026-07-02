"""
Team 3 - Phase 1: fusion model (verb + noun heads) with weighted loss,
plus ablation (text-only / object-only / fusion). Val-only evaluation.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

torch.manual_seed(0); np.random.seed(0)
F = "features"

def load(s): return {
    "text": np.load(f"{F}/{s}_text.npy"),
    "object": np.load(f"{F}/{s}_object.npy"),
    "verb": np.load(f"{F}/{s}_verb_class.npy"),
    "noun": np.load(f"{F}/{s}_noun_class.npy"),
}
tr, va = load("train"), load("val")
N_VERB = int(tr["verb"].max()) + 1
N_NOUN = int(tr["noun"].max()) + 1

def feats(d, mode):
    if mode == "text":   return d["text"]
    if mode == "object": return d["object"]
    return np.concatenate([d["text"], d["object"]], axis=1)  # fusion

def class_weights(y, n):
    cnt = np.bincount(y, minlength=n).astype(float)
    w = 1.0 / np.clip(cnt, 1, None)
    return torch.tensor(w / w.sum() * n, dtype=torch.float32)

class Net(nn.Module):
    def __init__(self, d_in, n_verb, n_noun):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3))
        self.verb = nn.Linear(128, n_verb)
        self.noun = nn.Linear(128, n_noun)
    def forward(self, x):
        h = self.trunk(x); return self.verb(h), self.noun(h)

def topk_acc(logits, y, k, mask):
    if mask.sum() == 0: return float("nan")
    tk = logits.topk(k, dim=1).indices.numpy()[mask]
    yy = y[mask]
    return float(np.mean([yy[i] in tk[i] for i in range(len(yy))]))

def run(mode, epochs=40):
    Xtr, Xva = feats(tr, mode).astype("float32"), feats(va, mode).astype("float32")
    vtr, ntr = torch.tensor(tr["verb"]), torch.tensor(tr["noun"])
    Xtr_t = torch.tensor(Xtr)
    wv, wn = class_weights(tr["verb"], N_VERB), class_weights(tr["noun"], N_NOUN)
    net = Net(Xtr.shape[1], N_VERB, N_NOUN)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lv, ln = nn.CrossEntropyLoss(weight=wv), nn.CrossEntropyLoss(weight=wn)
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            opt.zero_grad()
            pv, pn = net(Xtr_t[idx])
            loss = lv(pv, vtr[idx]) + ln(pn, ntr[idx])
            loss.backward(); opt.step()
    # eval
    net.eval()
    with torch.no_grad():
        pv, pn = net(torch.tensor(Xva))
    mv, mn = va["verb"] >= 0, va["noun"] >= 0
    pred_v, pred_n = pv.argmax(1).numpy(), pn.argmax(1).numpy()
    res = {
        "mode": mode,
        "verb_top1": round(topk_acc(pv, va["verb"], 1, mv), 4),
        "verb_top5": round(topk_acc(pv, va["verb"], 5, mv), 4),
        "verb_macroF1": round(f1_score(va["verb"][mv], pred_v[mv], average="macro", zero_division=0), 4),
        "noun_top1": round(topk_acc(pn, va["noun"], 1, mn), 4),
        "noun_top5": round(topk_acc(pn, va["noun"], 5, mn), 4),
        "noun_macroF1": round(f1_score(va["noun"][mn], pred_n[mn], average="macro", zero_division=0), 4),
    }
    m = mv & mn
    res["action_top1"] = round(float(np.mean((pred_v[m]==va["verb"][m]) & (pred_n[m]==va["noun"][m]))), 4)
    return res

results = [run(m) for m in ["text", "object", "fusion"]]
print(f"\n{'mode':8} {'verb@1':>7} {'verb@5':>7} {'verbF1':>7} {'noun@1':>7} {'noun@5':>7} {'nounF1':>7} {'act@1':>7}")
for r in results:
    print(f"{r['mode']:8} {r['verb_top1']:7.3f} {r['verb_top5']:7.3f} {r['verb_macroF1']:7.3f} "
          f"{r['noun_top1']:7.3f} {r['noun_top5']:7.3f} {r['noun_macroF1']:7.3f} {r['action_top1']:7.3f}")
json.dump(results, open("metrics.json", "w"), indent=2)
print("\nSaved metrics.json")
