# Week 2 Retrospective

## Finished
- Unified manifest with canonical label harmonization (`src/build_manifest.py`)
- Excluded BRRI out-of-scope `Rice` folder (16 images)
- Cross-dataset pHash dedup (`src/dedup.py`) with empty-collision fix
- Data loader with within-dataset and cross-dataset smoke tests (`src/data_loader.py`)
- Notes: inventory, label mapping, data rules, dedup report

## Key numbers
- Manifest rows: 5419
- BRRI Rice excluded: 16
- Cross-dataset duplicates (threshold 2): 0
- Smoke tests: both passed; class→index mapping identical (`brown_spot=0`, `tungro=1`)

## Edge cases found
1. BRRI `Rice` is not a leaf-disease class → excluded, not mapped.
2. Default Hamming threshold 5 produced false positives → tightened to 2.
3. Empty collision list crashed `sort_values` → fixed.

## Lesson
Verification matters more than defaults: a silent false-positive dedup or guessed label mapping would have contaminated Week 5 transfer numbers.

## Next (Week 3)
- Implement `src/train.py` and `src/eval.py`
- First end-to-end MobileNetV2 baseline on RiceLeafBD
- Do not start full matrix until that baseline lands in ~0.85–0.95 macro-F1
