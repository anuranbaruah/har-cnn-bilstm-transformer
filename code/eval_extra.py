"""Extra evaluation for the final report, run after all training is done.

For every saved checkpoint: re-measure CPU latency one model at a time (no other jobs running),
compute per-subject test accuracy, and list the most common confusions.
Then write summary.json with mean and std over seeds.

Usage: python eval_extra.py --data "<path>/UCI HAR Dataset" --res ../results
"""
import argparse, glob, json, os, time
import numpy as np
import torch
from data import load_har, CLASSES
from models import CNN1D, BiLSTMAttn, TransformerHAR

MODELS = {"cnn": CNN1D, "bilstm": BiLSTMAttn, "transformer": TransformerHAR}


@torch.no_grad()
def latency(model, batch=1, n=300):
    model.eval(); x = torch.randn(batch, 128, 9)
    for _ in range(30): model(x)
    ts = []
    for _ in range(n):
        t = time.perf_counter(); model(x); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000 / batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--res", default="../results")
    a = ap.parse_args()
    torch.set_num_threads(1)
    data = load_har(a.data)
    Xte, yte = data["test"]
    subj = np.loadtxt(os.path.join(a.data, "test", "subject_test.txt"), dtype=np.int64)
    X = torch.from_numpy(Xte)

    summary = {}
    for m, cls in MODELS.items():
        runs = []
        for f in sorted(glob.glob(os.path.join(a.res, f"{m}_seed*_results.json"))):
            r = json.load(open(f)); seed = int(f.split("seed")[1].split("_")[0])
            model = cls(); model.load_state_dict(torch.load(f.replace("_results.json", "_best.pt")))
            model.eval()
            with torch.no_grad():
                p = torch.cat([model(X[i:i + 512]).argmax(1) for i in range(0, len(X), 512)]).numpy()
            assert abs((p == yte).mean() - r["test"]["accuracy"]) < 1e-9, "checkpoint mismatch"
            per_subj = {int(s): float((p[subj == s] == yte[subj == s]).mean()) for s in np.unique(subj)}
            cm = np.array(r["test"]["confusion_matrix"])
            off = [(int(cm[i, j]), CLASSES[i], CLASSES[j]) for i in range(6) for j in range(6) if i != j]
            off.sort(reverse=True)
            runs.append({"seed": seed, "accuracy": r["test"]["accuracy"], "macro_precision": r["test"]["macro_precision"],
                         "macro_recall": r["test"]["macro_recall"], "macro_f1": r["test"]["macro_f1"],
                         "per_class_f1": r["test"]["per_class_f1"], "best_val_macro_f1": r["best_val_macro_f1"],
                         "epochs_run": r["epochs_run"], "train_time_s": r["train_time_s"],
                         "latency_ms_b1": latency(model, 1), "latency_ms_per_window_b256": latency(model, 256, 30),
                         "per_subject_acc": per_subj, "total_errors": int(cm.sum() - np.trace(cm)),
                         "sit_stand_errors": int(cm[3, 4] + cm[4, 3]), "top_confusions": off[:5]})
        if not runs: continue
        agg = lambda k: [float(np.mean([x[k] for x in runs])), float(np.std([x[k] for x in runs], ddof=1)) if len(runs) > 1 else 0.0]
        summary[m] = {"params": json.load(open(glob.glob(os.path.join(a.res, f"{m}_seed*_results.json"))[0]))["params"],
                      "n_seeds": len(runs),
                      **{k: agg(k) for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "best_val_macro_f1",
                                             "epochs_run", "train_time_s", "latency_ms_b1", "latency_ms_per_window_b256",
                                             "sit_stand_errors", "total_errors"]},
                      "per_class_f1_mean": {c: float(np.mean([x["per_class_f1"][c] for x in runs])) for c in CLASSES},
                      "runs": runs}
        print(m, {k: summary[m][k] for k in ["accuracy", "macro_f1", "latency_ms_b1"]}, flush=True)
    json.dump(summary, open(os.path.join(a.res, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
