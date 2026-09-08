# -*- coding: utf-8 -*-
"""Submission-readiness verification.

Complements the two existing audits rather than duplicating them:
  scripts/numerical_freeze_audit.py  -- recomputes stats from frozen CSVs
  scripts/audit_writing.py           -- manuscript numbers vs CSVs

This script checks what neither covers:
  (a) manuscript.md and the two .tex files agree on every headline number
  (b) all 13 figures are cited in each .tex
  (c) required sections exist, by their ACTUAL headings
No result is hardcoded; every line reflects a computed check.
"""
import io, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES  = ROOT.parent
MS   = ROOT / "paper/manuscript.md"
TEX  = {"arXiv": RES/"arXiv/main.tex", "PeerJ": RES/"PeerJ/peerj_manuscript.tex"}

rows = []
def chk(group, name, ok, detail=""):
    rows.append((group, name, bool(ok), detail))

# ---- (0) existing audits are authoritative; re-run them --------------------
for script, label in [("scripts/numerical_freeze_audit.py", "numerical audit"),
                      ("scripts/audit_writing.py", "writing audit")]:
    p = subprocess.run([sys.executable, script], cwd=ROOT,
                       capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip().splitlines()
    last = out[-1] if out else "(no output)"
    m = re.search(r"(\d+)\s*/\s*(\d+)\s+PASS", last)
    chk("audits", label, bool(m) and m.group(1) == m.group(2), last.split(" -> ")[0])

ms = io.open(MS, encoding="utf-8").read()

# ---- (1) headline numbers must appear in manuscript AND both .tex ----------
# three-seed primary family + seed-42 reference family + diagnosis/mitigation
HEADLINE = {
    "three-seed baseline 0.445": "0.445",
    "three-seed strong 0.502":   "0.502",
    "three-seed delta 0.063":    "0.063",
    "Wilcoxon W=26":             "26",
    "Wilcoxon p 0.0077":         "0.0077",
    "noise floor 0.057":         "0.057",
    "seed-42 baseline 0.436":    "0.436",
    "seed-42 delta 0.067":       "0.067",
    "seed-42 gap 0.387":         "0.387",
    "confound white 0.854":      "0.854",
    "confound field 0.705":      "0.705",
    "confound cross 0.573":      "0.573",
    "ablation geo 0.567":        "0.567",
    "ablation geo delta 0.085":  "0.085",
    "AdaBN mean -0.055":         "0.055",
}
for label, val in HEADLINE.items():
    in_ms = val in ms
    missing = [k for k, p in TEX.items()
               if val not in io.open(p, encoding="utf-8").read()]
    chk("numbers", label, in_ms and not missing,
        ("absent from manuscript; " if not in_ms else "") +
        ("absent from " + ",".join(missing) if missing else "all three agree"))

# ---- (1b) headline FRAMING: abstract/conclusion must lead three-seed --------
def slice_between(t, start, end):
    a = t.find(start)
    if a < 0:
        return ""
    b = t.find(end, a + len(start))
    return t[a: b if b > 0 else len(t)]

FRAME = {
    "arXiv": (r"\begin{abstract}", r"\end{abstract}", r"\section{Conclusion}"),
    "PeerJ": (r"\textbf{Background.}", r"\vspace{0.6em}", r"\section{Conclusions}"),
}
for tag, (a0, a1, c0) in FRAME.items():
    t = io.open(TEX[tag], encoding="utf-8").read()
    ci = t.find(c0)
    concl = slice_between(t[ci:], c0, r"\section{") if ci >= 0 else ""
    for part, seg in [("abstract", slice_between(t, a0, a1)),
                      ("conclusion", concl)]:
        leads = all(v in seg for v in ("0.445", "0.502", "0.063"))
        stale = [v for v in ("0.503", "0.387", "0.436") if v in seg]
        chk("framing", f"{tag} {part} leads three-seed", leads and not stale,
            ("three-seed present; " if leads else "three-seed INCOMPLETE; ") +
            ("seed-42 aggregate present: " + ",".join(stale) if stale
             else "no seed-42 aggregate"))

# ---- (2) every figure cited in each .tex -----------------------------------
figs = sorted(p.stem for p in (ROOT/"paper/figures").glob("fig*.png"))
for tag, path in TEX.items():
    t = io.open(path, encoding="utf-8").read()
    miss = [f for f in figs if f not in t]
    chk("figures", f"{tag}: all {len(figs)} figures included", not miss,
        "missing: " + ",".join(miss) if miss else f"{len(figs)}/{len(figs)} cited")

# ---- (3) required sections, by actual heading text ------------------------
for name, marker in [("Abstract","## Abstract"), ("Introduction","## 1. Introduction"),
                     ("Methods","## 4. Methods"), ("AI disclosure","### 4.9 Disclosure of AI assistance"),
                     ("Results","## 5. Results"), ("Discussion","## 6. Discussion"),
                     ("Future work","### 6.1 Future work"), ("Limitations","## 7. Limitations"),
                     ("Conclusion","## 8. Conclusion"), ("Reproducibility","## 9. Reproducibility"),
                     ("References","## References")]:
    chk("sections", name, marker in ms, marker)

# ---- report ----------------------------------------------------------------
GROUPS = ["audits", "numbers", "framing", "figures", "sections"]
unrendered = {r[0] for r in rows} - set(GROUPS)
assert not unrendered, f"check groups would be omitted from report: {unrendered}"
npass = sum(r[2] for r in rows)
lines = ["# Submission verification", "",
         f"**Result: {npass}/{len(rows)} checks PASS**", "",
         "Generated by `python scripts/verify_submission.py`. Complements "
         "`numerical_freeze_audit.py` (recomputes statistics) and `audit_writing.py` "
         "(manuscript vs CSVs); this file checks manuscript/LaTeX agreement, figure "
         "coverage, and section presence.", ""]
for grp in GROUPS:
    sub = [r for r in rows if r[0] == grp]
    if not sub: continue
    lines += [f"## {grp}", "", "| Check | Status | Detail |", "|---|---|---|"]
    lines += [f"| {n} | {'PASS' if ok else '**FAIL**'} | {d} |" for _, n, ok, d in sub]
    lines += [""]
out = ROOT/"notes/submission_verification.md"
io.open(out, "w", encoding="utf-8").write("\n".join(lines))
print(f"Submission verification: {npass}/{len(rows)} PASS -> {out}")
for _, n, ok, d in rows:
    if not ok: print("  FAIL:", n, "--", d)
