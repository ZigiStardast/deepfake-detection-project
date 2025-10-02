import os, argparse, pandas as pd, numpy as np, joblib
from tqdm import tqdm

def main(args):
    idx_csv = os.path.join(args.embeddings_dir, "embeddings_index.csv")
    df = pd.read_csv(idx_csv)

    if args.split:
        df = df[df["split"] == args.split]
        print(f"Filtering to split={args.split}, {len(df)} samples")

    print("Loading model:", args.model)
    model = joblib.load(args.model)

    out_rows = []
    for r in tqdm(df.itertuples(index=False), total=len(df)):
        embp = r.embedding_path
        if not os.path.exists(embp):
            continue
        emb = np.load(embp).reshape(1, -1)

        if hasattr(model, "predict_proba"):
            prob_fake = model.predict_proba(emb)[0][1]
        else:
            score = model.decision_function(emb).ravel()[0]
            prob_fake = 1 / (1 + np.exp(-score))  # sigmoid

        pred = "fake" if prob_fake >= 0.5 else "real"

        out_rows.append({
            "video_id": r.video_id,
            "frame_idx": int(r.frame_idx),
            "split": r.split,
            "true_label": r.label,
            "frame_path": r.frame_path,
            "prob_fake": float(prob_fake),
            "pred_label": pred
        })

    out_df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print("Saved predictions to", args.out_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_dir", required=True, help="Dir with embeddings + index csv")
    p.add_argument("--model", required=True, help="Path to .joblib classifier")
    p.add_argument("--out_csv", required=True, help="Where to save predictions")
    p.add_argument("--split", choices=["train","validation","test"], required=True,
                   help="Which split to predict (must be explicit)")
    args = p.parse_args()
    main(args)
