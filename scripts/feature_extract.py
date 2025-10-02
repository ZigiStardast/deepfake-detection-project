import os, argparse, re, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
import timm

IMG_EXTS = (".png", ".jpg", ".jpeg")

PATTERNS = [
    re.compile(r".*frame[_-]?(\d+)", re.IGNORECASE),   # frame_000123
    re.compile(r".*diff[_-]?(\d+)", re.IGNORECASE),    # diff_000123
    re.compile(r".*[_-](\d+)$"),                       # ..._123
]

def parse_frame_idx(fname_noext: str) -> int:
    for pat in PATTERNS:
        m = pat.match(fname_noext)
        if m:
            try:
                return int(m.group(1))
            except:
                pass
    digits = "".join(ch for ch in fname_noext if ch.isdigit())
    return int(digits) if digits else 0

def get_image_records(root_dir):
    """
    Vraća listu dict-ova:
      split, label, video_id, frame_idx, frame_path
    Radi i za:
      A) hijerarhiju .../<split>/<label>/<video_id>/<file.png>
      B) flat .../<split>/<label>/<file.png>  (video_id= ime fajla bez ekstenzije)
    """
    records = []
    for split in ("train", "validation", "test"):
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for label in ("fake", "real"):
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue

            for name in os.listdir(label_dir):
                path = os.path.join(label_dir, name)
                if os.path.isdir(path):
                    video_id = name
                    for f in os.listdir(path):
                        if f.lower().endswith(IMG_EXTS):
                            fp = os.path.join(path, f)
                            base = os.path.splitext(f)[0]
                            frame_idx = parse_frame_idx(base)
                            records.append({
                                "split": split,
                                "label": label,
                                "video_id": video_id,
                                "frame_idx": frame_idx,
                                "frame_path": fp
                            })
                else:
                    if name.lower().endswith(IMG_EXTS):
                        base = os.path.splitext(name)[0]
                        video_id = base
                        frame_idx = parse_frame_idx(base)  
                        fp = os.path.join(label_dir, name)
                        records.append({
                            "split": split,
                            "label": label,
                            "video_id": video_id,
                            "frame_idx": frame_idx,
                            "frame_path": fp
                        })

    records.sort(key=lambda r: (r["video_id"], r["frame_idx"]))
    return records

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = timm.create_model(args.model, pretrained=True, num_classes=0, global_pool="avg")
    model.eval().to(device)

    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    recs = get_image_records(args.input_dir)
    print(f"Found {len(recs)} images.")

    os.makedirs(args.output_dir, exist_ok=True)
    out_rows = []

    with torch.no_grad():
        for r in tqdm(recs):
            fp = r["frame_path"]
            try:
                img = Image.open(fp).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)
                feat = model(x).cpu().numpy().flatten()
            except Exception as e:
                print(f"[WARN] {e} - skipping {fp}")
                continue

            out_name = f"{r['video_id']}_{int(r['frame_idx']):06d}.npy"
            out_path = os.path.join(args.output_dir, out_name)
            np.save(out_path, feat)

            out_rows.append({
                "video_id": r["video_id"],
                "frame_idx": int(r["frame_idx"]),
                "split": r["split"],
                "label": r["label"],
                "frame_path": fp.replace("\\","/"),
                "embedding_path": out_path.replace("\\","/")
            })

    df = pd.DataFrame(out_rows)
    df.to_csv(os.path.join(args.output_dir, "embeddings_index.csv"), index=False)
    print("Saved embeddings to:", os.path.join(args.output_dir, "embeddings_index.csv"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50", help="Backbone (timm name)")
    p.add_argument("--input_dir", required=True, help="Root with frames_flat or frames_flat_diff_k1")
    p.add_argument("--output_dir", required=True, help="Where to save .npy features + index csv")
    p.add_argument("--img_size", type=int, default=224)
    args = p.parse_args()
    main(args)
