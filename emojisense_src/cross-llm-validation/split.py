# split.py
import json, random, unicodedata, os, argparse, hashlib

def nfc(s):
    """Normalize unicode to NFC (important for consistent emoji encoding)."""
    return unicodedata.normalize("NFC", s or "").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="path to filtered jsonl")
    ap.add_argument("--outdir", default="data", help="directory for train/valid output")
    ap.add_argument("--valid_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)

    data = []
    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            text = nfc(obj.get("original_text", ""))
            emoji = nfc(obj.get("emoji_sequence", ""))
            if not text or not emoji:
                continue

            # stable id (based on emoji+text)
            hid = hashlib.sha1((emoji + text).encode("utf-8")).hexdigest()[:12]
            data.append({"id": hid, "emoji": emoji, "text": text})

    rng.shuffle(data)
    n_valid = max(1, int(len(data) * args.valid_frac))
    valid, train = data[:n_valid], data[n_valid:]

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(os.path.join(args.outdir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.outdir, "valid.jsonl"), valid)

    print(f"Total examples: {len(data)}")
    print(f"Train: {len(train)}  |  Valid: {len(valid)}")

if __name__ == "__main__":
    main()
