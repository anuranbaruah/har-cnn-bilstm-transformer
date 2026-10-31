"""Common training and evaluation harness, identical for every model.

Usage: python train.py --model cnn|bilstm --data "<path to UCI HAR Dataset>" --out ../results
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from data import load_har, make_loaders, CLASSES
from models import CNN1D, BiLSTMAttn, TransformerHAR, count_params

MODELS = {"cnn": CNN1D, "bilstm": BiLSTMAttn, "transformer": TransformerHAR}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps = [], []
    for X, y in loader:
        ps.append(model(X).argmax(1)); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def metrics(y, p):
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    pc_f1 = precision_recall_fscore_support(y, p, average=None, zero_division=0)[2]
    return {"accuracy": accuracy_score(y, p), "macro_precision": prec, "macro_recall": rec,
            "macro_f1": f1, "per_class_f1": dict(zip(CLASSES, pc_f1.tolist())),
            "confusion_matrix": confusion_matrix(y, p).tolist()}


@torch.no_grad()
def latency_ms(model, n=200):
    model.eval()
    x = torch.randn(1, 128, 9)
    for _ in range(20): model(x)
    t = time.perf_counter()
    for _ in range(n): model(x)
    return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    set_seed(a.seed)
    data = load_har(a.data)
    loaders = make_loaders(data, seed=a.seed)
    model = MODELS[a.model]()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    history, best_f1, best_state, wait = [], -1, None, 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); total = 0.0
        for X, y in loaders["train"]:
            opt.zero_grad(); loss = loss_fn(model(X), y); loss.backward(); opt.step()
            total += loss.item() * len(y)
        yv, pv = predict(model, loaders["val"])
        vm = metrics(yv, pv)
        history.append({"epoch": ep, "train_loss": total / len(loaders["train"].dataset),
                        "val_acc": vm["accuracy"], "val_f1": vm["macro_f1"]})
        print(f"[{a.model}] epoch {ep:2d} loss {history[-1]['train_loss']:.4f} "
              f"val_acc {vm['accuracy']:.4f} val_f1 {vm['macro_f1']:.4f}", flush=True)
        if vm["macro_f1"] > best_f1:
            best_f1, wait = vm["macro_f1"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= a.patience: break
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    yt, pt = predict(model, loaders["test"])
    res = {"model": a.model, "params": count_params(model), "epochs_run": len(history),
           "best_val_macro_f1": best_f1, "train_time_s": train_time,
           "cpu_latency_ms_per_window": latency_ms(model),
           "test": metrics(yt, pt), "history": history,
           "split_sizes": {k: int(len(data[k][1])) for k in ["train", "val", "test"]}}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.model}_seed{a.seed}_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    torch.save(best_state, os.path.join(a.out, f"{a.model}_seed{a.seed}_best.pt"))
    print(json.dumps({k: res[k] for k in ["params", "epochs_run", "train_time_s"]}),
          json.dumps({k: v for k, v in res["test"].items() if k != "confusion_matrix"}, indent=1))


if __name__ == "__main__":
    main()
