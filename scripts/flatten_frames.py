import argparse, os, shutil
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset_csv)
    for _, row in df.iterrows():
        vid = row["video_id"]
        label = row["label"]
        split = row["split"]
        src_dir = Path(row["video_path"])
        dst_dir = Path(args.out_root) / split / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img in src_dir.glob("*.jpg"):
            # ime fajla = videoid_frame.jpg
            new_name = f"{vid}_{img.name}"
            shutil.copy(img, dst_dir / new_name)

if __name__ == "__main__":
    main()
