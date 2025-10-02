import argparse, glob, os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

def load_frames(feats_dir):
    rows = []
    for path in glob.glob(os.path.join(feats_dir, "*.npz")):
        d = np.load(path, allow_pickle=True)
        feats = d["features"]          
        label = 1 if str(d["label"]) == "fake" else 0
        vid   = str(d["video_id"])
        for i in range(feats.shape[0]):
            rows.append((vid, label, feats[i]))
    vids, ys, xs = zip(*rows)
    X = np.vstack(xs); y = np.array(ys); vids = np.array(vids)
    return X, y, vids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats_dir", required=True)
    ap.add_argument("--out_csv", default="results/frame_results.csv")
    args = ap.parse_args()

    X, y, vids = load_frames(args.feats_dir)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    clf = Pipeline([("scaler", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000))])
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:,1]
    pred = (prob >= 0.5).astype(int)

    acc = accuracy_score(yte, pred)
    f1  = f1_score(yte, pred)
    auc = roc_auc_score(yte, prob)
    print(classification_report(yte, pred))
    pd.DataFrame([{"model":"frame_logreg","accuracy":acc,"f1":f1,"auc":auc}]).to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)

if __name__ == "__main__":
    main()
