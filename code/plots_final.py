"""Figures for the final report. Reads results/*_seed*_results.json and results/summary.json.

Usage: python plots_final.py ../results
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = sys.argv[1] if len(sys.argv) > 1 else "../results"
NAMES = {"cnn": "1D-CNN", "bilstm": "BiLSTM", "transformer": "Transformer"}
COLORS = {"cnn": "#2a78d6", "bilstm": "#eb6834", "transformer": "#1baf7a"}
SHORT = ["Walk", "Up", "Down", "Sit", "Stand", "Lay"]
FULL = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"], "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
S = json.load(open(os.path.join(res, "summary.json")))
models = [m for m in NAMES if m in S]
seed42 = {m: json.load(open(os.path.join(res, f"{m}_seed42_results.json"))) for m in models}

# Fig: confusion matrices (seed 42)
fig, axes = plt.subplots(1, len(models), figsize=(7.2, 2.6))
for ax, m in zip(axes, models):
    cm = np.array(seed42[m]["test"]["confusion_matrix"])
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6.5,
                    color="white" if cm[i, j] / cm[i].sum() > 0.5 else "#333")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(SHORT, fontsize=6.5); ax.set_yticklabels(SHORT, fontsize=6.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{NAMES[m]} (test, seed 42)", fontsize=8)
    ax.spines[:].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(res, "fig_confusion_matrices.png"), dpi=220); plt.close()

# Fig: training curves (seed 42)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
for m in models:
    h = seed42[m]["history"]; ep = [x["epoch"] for x in h]
    axes[0].plot(ep, [x["train_loss"] for x in h], color=COLORS[m], lw=1.6, label=NAMES[m])
    axes[1].plot(ep, [x["val_f1"] for x in h], color=COLORS[m], lw=1.6, label=NAMES[m])
axes[0].set_title("Training loss (seed 42)"); axes[1].set_title("Validation macro F1 (seed 42)")
for ax in axes:
    ax.set_xlabel("Epoch"); ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(res, "fig_training_curves.png"), dpi=220); plt.close()

# Fig: per-class F1, mean over seeds
fig, ax = plt.subplots(figsize=(7.2, 2.4))
w = 0.8 / len(models); x = np.arange(6)
for k, m in enumerate(models):
    vals = [S[m]["per_class_f1_mean"][c] for c in FULL]
    ax.bar(x + (k - (len(models) - 1) / 2) * w, vals, w * 0.92, color=COLORS[m], label=NAMES[m])
ax.set_xticks(x); ax.set_xticklabels(FULL); ax.set_ylim(0.6, 1.0); ax.set_ylabel("Test F1 (mean of 3 seeds)")
ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0))
plt.tight_layout(); plt.savefig(os.path.join(res, "fig_per_class_f1.png"), dpi=220); plt.close()

# Fig: per-subject test accuracy (mean over seeds)
subs = sorted(S[models[0]]["runs"][0]["per_subject_acc"], key=int)
fig, ax = plt.subplots(figsize=(7.2, 2.3))
w = 0.8 / len(models); x = np.arange(len(subs))
for k, m in enumerate(models):
    vals = [np.mean([r["per_subject_acc"][s] for r in S[m]["runs"]]) for s in subs]
    ax.bar(x + (k - (len(models) - 1) / 2) * w, vals, w * 0.92, color=COLORS[m], label=NAMES[m])
ax.set_xticks(x); ax.set_xticklabels([f"S{s}" for s in subs]); ax.set_ylim(0.6, 1.0)
ax.set_ylabel("Test accuracy (mean of 3 seeds)"); ax.set_xlabel("Test subject")
ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0))
plt.tight_layout(); plt.savefig(os.path.join(res, "fig_per_subject_acc.png"), dpi=220); plt.close()
print("saved figures")
