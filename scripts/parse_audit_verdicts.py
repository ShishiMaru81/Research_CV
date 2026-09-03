"""Parse hand-filled mask audit sheet and print Week 12 decision gate."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT_CSV = ROOT / "notes" / "mask_audit" / "audit_sheet.csv"
OUT_MD = ROOT / "notes" / "mask_audit" / "audit_decision.md"
ACCEPTABLE = {"PASS", "PARTIAL"}
VALID_VERDICTS = {"PASS", "PARTIAL", "FAIL"}
DATASETS = ("riceleafbd", "dhan_shomadhan", "brri_rice_disease_pest")
GATE = 0.80


def main() -> None:
    if not AUDIT_CSV.is_file():
        raise FileNotFoundError(f"Missing audit sheet: {AUDIT_CSV}")

    audit = pd.read_csv(AUDIT_CSV)
    print(f"Parsed {AUDIT_CSV} ({len(audit)} rows)")
    print(f"Columns: {list(audit.columns)}")

    required = [
        "index",
        "image_path",
        "dataset",
        "mapped_class",
        "sam_verdict",
        "hsv_verdict",
        "reason_code",
        "notes",
    ]
    missing_cols = [c for c in required if c not in audit.columns]
    if missing_cols:
        raise KeyError(f"audit_sheet missing columns: {missing_cols}")

    for col in ("sam_verdict", "hsv_verdict"):
        empty = audit[col].isna() | (audit[col].astype(str).str.strip() == "")
        if empty.any():
            bad = audit.loc[empty, "index"].tolist()
            raise ValueError(
                f"{col} has unfilled rows at index={bad}. "
                "Fill all verdicts before running the gate."
            )
        audit[col] = audit[col].astype(str).str.strip().str.upper()
        invalid = set(audit[col].unique()) - VALID_VERDICTS
        if invalid:
            raise ValueError(
                f"{col} has invalid values {invalid}. "
                f"Allowed: {sorted(VALID_VERDICTS)}"
            )

    print("\nAcceptance rates by variant and dataset:")
    rates: dict[str, dict[str, tuple[int, int, float]]] = {
        "sam_leaf": {},
        "hsv_leaf": {},
    }

    for variant, col in (("sam_leaf", "sam_verdict"), ("hsv_leaf", "hsv_verdict")):
        print(f"\n{variant}:")
        for dataset in DATASETS:
            sub = audit[audit["dataset"] == dataset]
            if len(sub) == 0:
                raise ValueError(f"No audit rows for dataset={dataset}")
            ok = sub[col].isin(ACCEPTABLE).sum()
            total = len(sub)
            pct = ok / total
            rates[variant][dataset] = (int(ok), int(total), float(pct))
            mark = "PASS" if pct >= GATE else "FAIL"
            print(f"  {dataset:24} {ok:2}/{total:2} = {pct:.1%}  {mark}")

    print("\nSummary table:")
    print("Variant     | riceleafbd | dhan_shomadhan | brri")
    for variant in ("sam_leaf", "hsv_leaf"):
        cells = []
        for dataset in DATASETS:
            ok, total, _ = rates[variant][dataset]
            cells.append(f"{ok}/{total}")
        print(f"{variant:11} | {cells[0]:10} | {cells[1]:14} | {cells[2]}")

    print("\nDECISION GATE:")
    cleared: list[str] = []
    lines = [
        "# Mask audit decision (Week 12)",
        "",
        f"Source: `{AUDIT_CSV.as_posix()}`",
        f"Gate threshold: acceptable (PASS+PARTIAL) >= {GATE:.0%} per dataset.",
        "",
    ]
    for variant in ("sam_leaf", "hsv_leaf"):
        weakest = min(rates[variant].items(), key=lambda kv: kv[1][2])
        weakest_ds, (ok, total, pct) = weakest
        if pct >= GATE:
            cleared.append(variant)
            msg = f"PASS {variant} (min {pct:.1%} on {weakest_ds}: {ok}/{total})"
            print(f"  PASS  {variant:12} (min {pct:.1%} on {weakest_ds})")
        else:
            msg = f"FAIL {variant} (min {pct:.1%} on {weakest_ds}: {ok}/{total})"
            print(f"  FAIL  {variant:12} (min {pct:.1%} on {weakest_ds})")
            print("        -> Do NOT train Week 13 with this variant")
        lines.append(f"- {msg}")

    lines.append("")
    if not cleared:
        lines.append(
            "**Week 13 blocked:** both variants failed the gate. "
            "Do not proceed to masked retraining."
        )
        print("\nWeek 13 instructions: BLOCKED — both variants failed.")
    else:
        lines.append(
            f"**Variants cleared for Week 13:** {', '.join(cleared)}"
        )
        dropped = [v for v in ("sam_leaf", "hsv_leaf") if v not in cleared]
        if dropped:
            lines.append(
                f"**Dropped:** {', '.join(dropped)} — note in Limitations."
            )
        print(f"\nWeek 13 instructions: run with {cleared} only.")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_MD}")
    print(f"Shape: {audit.shape}")
    print(audit.head())


if __name__ == "__main__":
    main()
