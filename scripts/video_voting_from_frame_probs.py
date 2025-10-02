import os, argparse, pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def majority_vote(group):
    preds = group["pred_label"].values
    return np.bincount((preds=="fake").astype(int)).argmax()

def avg_prob_vote(group):
    return 1 if group["prob_fake"].mean() >= 0.5 else 0

def weighted_vote(group):
    weights = np.abs(group["prob_fake"] - 0.5)  # confidence = udaljenost od 0.5
    if weights.sum() == 0:
        return avg_prob_vote(group)
    weighted = np.average((group["prob_fake"]>=0.5).astype(int), weights=weights)
    return 1 if weighted >= 0.5 else 0

def evaluate_video_preds(df):
    y_true = (df["true_label"]=="fake").astype(int)
    y_pred = df["video_pred"]
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    try:
        auc = roc_auc_score(y_true, df["video_prob"])
    except:
        auc = None
    return acc, prec, rec, f1, auc

def main(args):
    df = pd.read_csv(args.frame_preds)
    print("Loaded frame preds:", df.shape)

    results = []
    grouped = df.groupby("video_id")

    strategies = {
        "majority": majority_vote,
        "avgprob": avg_prob_vote,
        "weighted": weighted_vote
    }

    for strat, func in strategies.items():
        video_preds = grouped.apply(func)

        df_vid = pd.DataFrame({
            "video_id": video_preds.index,
            "video_pred": video_preds.values,
            "true_label": grouped["true_label"].agg("first").values,
            "video_prob": grouped["prob_fake"].mean().values
        })

        acc,prec,rec,f1,auc = evaluate_video_preds(df_vid)
        results.append((strat, acc,prec,rec,f1,auc))

        out_path = args.out.replace(".csv", f"_{strat}.csv")
        df_vid.to_csv(out_path, index=False)
        print(f"Saved {strat} predictions -> {out_path}")

    print("\n=== Video-level results ===")
    for strat,acc,prec,rec,f1,auc in results:
        print(f"{strat:10s} | Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} "
              f"F1={f1:.3f} AUC={auc:.3f}")

    df_res = pd.DataFrame(results, columns=["strategy","acc","prec","rec","f1","auc"])
    df_res.to_csv(args.out.replace(".csv","_summary.csv"), index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frame_preds", required=True, help="Path to frame-level CSV")
    p.add_argument("--out", required=True, help="Base path to save video-level CSVs")
    args = p.parse_args()
    main(args)
