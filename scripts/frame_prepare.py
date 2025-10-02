# scripts/frame_prepare.py
import os, cv2, pandas as pd, numpy as np, argparse
from tqdm import tqdm

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_resized(path, w,h):
    img = cv2.imread(path)
    if img is None: raise IOError(f"Can't read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (w,h))

def save_rgb(img, out_path):
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def main(a):
    df = pd.read_csv(a.csv).sort_values(["video_id","frame_idx"]).reset_index(drop=True)
    cache = {}

    for _, r in tqdm(df.iterrows(), total=len(df)):
        vid = str(r["video_id"]); idx = int(r["frame_idx"])
        split = r["split"]; label = r["label"]; p = r["video_path"].replace("\\","/")

        try:
            img = load_resized(p, a.resize_w, a.resize_h)
        except Exception as e:
            print(f"[WARN] {e}"); continue

        out_dir = os.path.join(a.out_dir, split, label, vid)
        if a.make_diffs:
            last = cache.get(vid)
            if last is not None and (idx - last[0]) == a.diff_k:
                diff = np.abs(img.astype(np.int16) - last[1].astype(np.int16)).astype(np.uint8)
                save_rgb(diff, os.path.join(out_dir, f"diff_{idx:06d}.png"))
            cache[vid] = (idx, img)
        else:
            save_rgb(img, os.path.join(out_dir, f"frame_{idx:06d}.png"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--resize_w", type=int, default=224)
    p.add_argument("--resize_h", type=int, default=224)
    p.add_argument("--make_diffs", action="store_true")
    p.add_argument("--diff_k", type=int, default=1)
    a = p.parse_args()
    main(a)
