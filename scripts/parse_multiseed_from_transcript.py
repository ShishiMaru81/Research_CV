"""Recover multi-seed RECORDED metrics from the Cursor agent transcript."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

TRANSCRIPT = Path(
    r"C:\Users\paula\.cursor\projects\d-Research-Own\agent-transcripts"
    r"\27d0d383-1806-41f5-bd8a-dfe51c688375\27d0d383-1806-41f5-bd8a-dfe51c688375.jsonl"
)

transfer_pat = re.compile(
    r"RECORDED:\s+(\S+)\s+\|\s+([^|]+)\s+\|\s+same=([\d.]+)\s+\|\s+cross=([\d.]+)\s+\|\s+gap=([\d.]+)"
)
lodo_pat = re.compile(
    r"RECORDED:\s+(\S+)\s+\|\s+holdout=(\S+)\s+\|\s+macro_f1=([\d.]+)\s+\|\s+n=(\d+)"
)
train_xfer = re.compile(r"=== TRAIN (\S+): (\S+) -> (\S+) \(seed=(\d+)\) ===")
train_lodo = re.compile(r"=== TRAIN (\S+): .+ \(holdout (\S+), seed=(\d+)\) ===")


def parse_message(text: str) -> tuple[list[dict], list[dict]]:
    xfer: list[dict] = []
    lodo: list[dict] = []
    current_aug = "default"
    pending_train = None
    for line in text.splitlines():
        if "--- transfer_baseline ---" in line:
            current_aug = "default"
        elif "--- transfer_aug" in line:
            current_aug = "strong"
        m = train_xfer.search(line)
        if m:
            pending_train = (
                "xfer",
                m.group(1),
                m.group(2),
                m.group(3),
                int(m.group(4)),
                current_aug,
            )
            continue
        m = train_lodo.search(line)
        if m:
            pending_train = ("lodo", m.group(1), m.group(2), int(m.group(3)))
            continue
        m = transfer_pat.search(line)
        if m:
            model = m.group(1)
            pair = m.group(2).strip()
            same, cross, gap = float(m.group(3)), float(m.group(4)), float(m.group(5))
            if ":" in pair:
                tr, ev = pair.split(":", 1)
            else:
                tr, ev = None, None
            seed = None
            aug = current_aug
            if pending_train and pending_train[0] == "xfer":
                model, tr, ev, seed, aug = (
                    pending_train[1],
                    pending_train[2],
                    pending_train[3],
                    pending_train[4],
                    pending_train[5],
                )
            xfer.append(
                {
                    "model": model,
                    "train_dataset": tr,
                    "test_dataset": ev,
                    "seed": seed,
                    "augmentation": aug,
                    "in_macro_f1": same,
                    "cross_macro_f1": cross,
                    "gap": gap,
                }
            )
            continue
        m = lodo_pat.search(line)
        if m:
            model, holdout = m.group(1), m.group(2)
            f1, n = float(m.group(3)), int(m.group(4))
            seed = None
            if pending_train and pending_train[0] == "lodo":
                model, holdout, seed = pending_train[1], pending_train[2], pending_train[3]
            lodo.append(
                {
                    "model": model,
                    "held_out_dataset": holdout,
                    "seed": seed,
                    "macro_f1": f1,
                    "n_samples": n,
                }
            )
    return xfer, lodo


def main() -> None:
    candidates: list[tuple[int, int, str]] = []
    with TRANSCRIPT.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("role") != "user":
                continue
            content = obj.get("message", {}).get("content", [])
            texts = [
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            text = "\n".join(texts)
            n_rec = text.count("RECORDED:")
            if n_rec and ("multi-seed" in text or "train_seed=" in text or "seed=7" in text or "seed=2024" in text):
                candidates.append((n_rec, i, text))

    print(f"Candidate messages: {len(candidates)}; counts={[c[0] for c in candidates]}")

    all_xfer: list[dict] = []
    all_lodo: list[dict] = []
    for _n, _i, text in sorted(candidates, key=lambda x: x[1]):
        x, l = parse_message(text)
        all_xfer.extend(x)
        all_lodo.extend(l)

    xf = pd.DataFrame(all_xfer)
    ld = pd.DataFrame(all_lodo)
    if len(xf):
        xf = xf.drop_duplicates(
            subset=["model", "train_dataset", "test_dataset", "seed", "augmentation"],
            keep="last",
        )
    if len(ld):
        ld = ld.drop_duplicates(subset=["model", "held_out_dataset", "seed"], keep="last")

    print("xfer:", len(xf))
    if len(xf):
        print(xf.groupby(["seed", "augmentation"]).size())
    print("lodo:", len(ld))
    if len(ld):
        print(ld.groupby("seed").size())

    out = Path("results/multiseed")
    out.mkdir(parents=True, exist_ok=True)
    xf.to_csv(out / "_parsed_from_logs_transfer.csv", index=False)
    ld.to_csv(out / "_parsed_from_logs_lodo.csv", index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
