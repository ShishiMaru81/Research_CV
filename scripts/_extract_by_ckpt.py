"""Extract seed-2024 strong-aug RECORDED metrics via checkpoint path cues."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

TRANSCRIPT = Path(
    r"C:\Users\paula\.cursor\projects\d-Research-Own\agent-transcripts"
    r"\27d0d383-1806-41f5-bd8a-dfe51c688375\27d0d383-1806-41f5-bd8a-dfe51c688375.jsonl"
)

# Match a training block ending in RECORDED, keyed by checkpoint path
block_pat = re.compile(
    r"=== TRAIN (?P<model>\S+): (?P<tr>\S+) -> (?P<te>\S+) \(seed=(?P<seed>\d+)\) ==="
    r"(?P<body>.*?)"
    r"RECORDED:.*?same=(?P<same>[\d.]+)\s+\|\s+cross=(?P<cross>[\d.]+)\s+\|\s+gap=(?P<gap>[\d.]+)",
    re.S,
)
ckpt_pat = re.compile(r"Best checkpoint:\s*(?P<path>\S+)")


def main() -> None:
    rows = []
    with TRANSCRIPT.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("role") != "user":
                continue
            content = obj.get("message", {}).get("content", [])
            text = "\n".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
            for m in block_pat.finditer(text):
                body = m.group("body")
                ck = ckpt_pat.search(body)
                path = ck.group("path") if ck else ""
                aug = "strong" if "aug-strong" in path else "default"
                if "lodo-holdout" in path:
                    continue
                rows.append(
                    {
                        "model": m.group("model"),
                        "train_dataset": m.group("tr"),
                        "test_dataset": m.group("te"),
                        "seed": int(m.group("seed")),
                        "augmentation": aug,
                        "in_macro_f1": float(m.group("same")),
                        "cross_macro_f1": float(m.group("cross")),
                        "gap": float(m.group("gap")),
                        "checkpoint_path": path,
                    }
                )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(
        subset=["model", "train_dataset", "test_dataset", "seed", "augmentation"],
        keep="last",
    )
    print(df.groupby(["seed", "augmentation"]).size())
    print("\nseed 2024 strong:")
    s = df[(df.seed == 2024) & (df.augmentation == "strong")].sort_values(
        ["train_dataset", "test_dataset", "model"]
    )
    print(s.to_string(index=False))
    print("count", len(s))
    out = Path("results/multiseed/_ckpt_cued_transfer.csv")
    df.to_csv(out, index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
