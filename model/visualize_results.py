"""Team 3 - generate presentation plots."""
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib
matplotlib.rcParams.update({"font.size":12,"figure.dpi":150})

# ── 1. Ablation bar chart ────────────────────────────────────────────────────
raw = [
    {"mode":"Text only",    "verb@1":0.973,"noun@1":0.967,"action@1":0.944,"verbF1":0.787,"nounF1":0.831},
    {"mode":"Object only",  "verb@1":0.122,"noun@1":1.000,"action@1":0.129,"verbF1":0.085,"nounF1":1.000},
    {"mode":"Fusion\n(text+obj)","verb@1":0.982,"noun@1":1.000,"action@1":0.986,"verbF1":0.918,"nounF1":1.000},
    {"mode":"Video only\n(P01_01)",   "verb@1":0.059,"noun@1":0.042,"action@1":0.000,"verbF1":0.003,"nounF1":0.001},
]
df = pd.DataFrame(raw)
fig, axes = plt.subplots(1,2, figsize=(13,5))
fig.suptitle("Ablation Study — Val Set (EPIC-KITCHENS P01)", fontsize=14, fontweight="bold")

x = np.arange(len(df)); w = 0.28
ax = axes[0]
ax.bar(x-w, df["verb@1"],   w, label="Verb Top-1",   color="#4C72B0")
ax.bar(x,   df["noun@1"],   w, label="Noun Top-1",   color="#DD8452")
ax.bar(x+w, df["action@1"], w, label="Action Top-1", color="#55A868")
ax.set_xticks(x); ax.set_xticklabels(df["mode"], fontsize=10)
ax.set_ylabel("Accuracy"); ax.set_ylim(0,1.12); ax.set_title("Top-1 Accuracy")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
for bar in ax.patches: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)

ax2 = axes[1]
ax2.bar(x-w/2, df["verbF1"], w, label="Verb macro-F1", color="#4C72B0")
ax2.bar(x+w/2, df["nounF1"], w, label="Noun macro-F1", color="#DD8452")
ax2.set_xticks(x); ax2.set_xticklabels(df["mode"], fontsize=10)
ax2.set_ylabel("Macro-F1"); ax2.set_ylim(0,1.12); ax2.set_title("Macro-F1 (penalises tail classes)")
ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
for bar in ax2.patches: ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)

plt.tight_layout(); plt.savefig("plot_ablation.png", bbox_inches="tight"); plt.close()
print("Saved plot_ablation.png")

# ── 2. Top-verb distribution + model coverage ────────────────────────────────
train = pd.read_csv("../data/train_metadata.csv")
preds = pd.read_csv("test_predictions.csv")
top_verbs = train["verb"].value_counts().head(12)

fig, axes = plt.subplots(1,2, figsize=(13,5))
fig.suptitle("Class Distribution & Prediction Coverage", fontsize=14, fontweight="bold")

axes[0].barh(top_verbs.index[::-1], top_verbs.values[::-1], color="#4C72B0", alpha=0.8)
axes[0].set_xlabel("Train samples"); axes[0].set_title("Top-12 Verb Distribution (train)")
axes[0].grid(axis="x", alpha=0.3)

pred_verb_counts = preds["pred_verb"].value_counts()
axes[1].barh(pred_verb_counts.index[::-1], pred_verb_counts.values[::-1], color="#55A868", alpha=0.8)
axes[1].set_xlabel("Predicted count"); axes[1].set_title("Predicted Verb Distribution (test, 647 rows)")
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout(); plt.savefig("plot_distributions.png", bbox_inches="tight"); plt.close()
print("Saved plot_distributions.png")

# ── 3. Results table as image ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11,3))
ax.axis("off")
cols = ["Model","Verb@1","Verb@5","Verb F1","Noun@1","Noun@5","Noun F1","Action@1"]
rows = [
    ["Text only",        "0.973","0.998","0.787","0.967","0.990","0.831","0.944"],
    ["Object only",      "0.122","0.704","0.085","1.000","1.000","1.000","0.129"],
    ["Fusion (text+obj)","0.982","1.000","0.918","1.000","1.000","1.000","0.986"],
    ["Video only*",      "0.059","0.595","0.003","0.042","0.053","0.001","0.000"],
]
colors = [["#e8f0fe"]*8 if i%2==0 else ["#ffffff"]*8 for i in range(len(rows))]
t = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center",
             cellColours=colors)
t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,1.8)
for j in range(len(cols)): t[(0,j)].set_facecolor("#2c5f8a"); t[(0,j)].get_text().set_color("white")
ax.set_title("Team 3 — Full Results Table\n* Video-only trained on 329 P01_01 clips only (1 of 22 train videos)",
             fontsize=11, pad=10)
plt.tight_layout(); plt.savefig("plot_results_table.png", bbox_inches="tight", facecolor="white"); plt.close()
print("Saved plot_results_table.png")
print("\nAll plots done.")
