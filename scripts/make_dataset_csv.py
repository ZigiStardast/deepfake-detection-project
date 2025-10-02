import os, csv, re

ROOT = "faceforensics_root"            
SPLITS = ["train", "validation", "test"]
LABELS = ["real", "fake"]
OUT_CSV = "data/dataset.csv"

# Regex:
#  - case A: ..._<digits>.png  -> video_id = sve pre poslednjeg _<digits>, frame_idx = taj broj
#  - case B: bez trailing _<digits> (npr '000_003.png') -> tretiraj kao single frame (idx=0)
TRAILING_NUM = re.compile(r"^(.*)_(\d+)$")
ONLY_TWO_3DIG = re.compile(r"^\d{3}_\d{3}$")  # specijalan slučaj (npr. 000_003)

def parse_name(name_noext):
    m = TRAILING_NUM.match(name_noext)
    if m:
        video_id = m.group(1)
        frame_idx = int(m.group(2))
        return video_id, frame_idx
    if ONLY_TWO_3DIG.match(name_noext):
        return name_noext, 0
    return name_noext, 0

def main():
    rows = []
    for split in SPLITS:
        for label in LABELS:
            folder = os.path.join(ROOT, split, label)
            if not os.path.isdir(folder):
                continue
            for f in os.listdir(folder):
                if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                name = os.path.splitext(f)[0]
                video_id, frame_idx = parse_name(name)
                rel_path = os.path.join(ROOT, split, label, f).replace("\\", "/")
                rows.append([video_id, frame_idx, label, split, rel_path])

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["video_id", "frame_idx", "label", "split", "video_path"])
        rows.sort(key=lambda r: (r[0], r[1]))
        w.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
