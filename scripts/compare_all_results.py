import argparse
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def collect_metrics(video_csv: Path):
    try:
        df = pd.read_csv(video_csv)
    except Exception as e:
        print(f"[WARN] Cannot read {video_csv}: {e}")
        return None
    if "video_pred" not in df.columns or "true_label" not in df.columns:
        return None

    y_true = (df["true_label"] == "fake").astype(int)
    y_pred = df["video_pred"].astype(int)
    prob = df.get("video_prob", pd.Series([0.5]*len(df)))  # fallback

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    try:
        auc = roc_auc_score(y_true, prob)
    except Exception:
        auc = float("nan")

    return acc, prec, rec, f1, auc, len(df)

def parse_filename(fname: str):
    """
    Expected format: video_preds_<model>_<split>_<strategy>.csv
    Example: video_preds_resnet50std_xgb_val_avgprob.csv
    """
    parts = fname.replace(".csv", "").split("_")
    if len(parts) < 5:
        return None, None, None
    model = parts[2]        # resnet50std or effb3diffk1
    split = parts[3]        # val/test
    strategy = parts[-1]    # majority/avgprob/weighted
    return model, split, strategy

def main(args):
    preds_dir = Path(args.preds_dir)
    rows = []
    for f in preds_dir.glob("video_preds*.csv"):
        metrics = collect_metrics(f)
        if metrics is None:
            continue
        acc, prec, rec, f1, auc, n = metrics
        model, split, strat = parse_filename(f.name)
        rows.append({
            "file": f.name,
            "model": model,
            "split": split,
            "strategy": strat,
            "acc": round(acc, 3),
            "prec": round(prec, 3),
            "rec": round(rec, 3),
            "f1": round(f1, 3),
            "auc": round(auc, 3) if auc == auc else "nan",
            "videos": n
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid prediction CSVs found.")
        return

    df = df.sort_values(by="f1", ascending=False)
    out_csv = preds_dir / "summary_metrics.csv"
    df.to_csv(out_csv, index=False)
    print("Saved summary table to", out_csv)
    print(df.to_string(index=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preds_dir", default="preds", help="Directory with video_preds_*.csv files")
    args = p.parse_args()
    main(args)
