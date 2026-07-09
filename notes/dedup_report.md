# Dedup Report (Week 2)

## Method
- Perceptual hash: `imagehash.phash`, `hash_size=8`
- Cross-dataset only (same-dataset pairs ignored)
- BK-tree search for near-neighbor matching

## Threshold decision
| Threshold | Pairs found | Hand-check result | Decision |
|-----------|-------------|-------------------|----------|
| 5 (default) | 5 | False positives: similar white-background leaf photos with different lesions/classes | Reject |
| 2 | 0 | No pairs | **Accepted** |

## Final setting
- **Hamming threshold = 2**
- Total cross-dataset duplicate pairs: **0**
- Manifest rows flagged `is_duplicate=True`: **0**
- Manifest size: **5419** images (after excluding BRRI `Rice` 16 images)

## Interpretation
At a sensible threshold, there is no evidence of true cross-dataset image copies among the three datasets. The threshold-5 hits were compositionally similar studio/white-bg leaves, not identical photos. This is documented as a limitation note: near-duplicate risk appears low for this collection.

## Outputs
- `results/manifest.csv` (updated `is_duplicate` column)
- `results/duplicates_report.csv` (empty under threshold 2)
