# Data Rules (Non-negotiable)

1. Originals only
- Keep original images only.
- Exclude any augmented-image folders/files (especially BRRI archive's Augmented Dataset).

2. Cross-dataset de-duplication
- Detect near-duplicates across datasets with perceptual hashing in Week 2.
- Flag and exclude cross-copies from evaluation splits.

3. Frozen test split
- Create seeded train/val/test splits in Week 2.
- Freeze test split immediately after creation; no re-splitting during model experiments.

4. Label integrity
- Use canonical mapping only; do not guess unknown labels.
- Any unmatched raw class must halt manifest generation for explicit mapping updates.
- Explicit exclusions (documented, not guessed): BRRI `Rice` folder (16 images) is out-of-scope and excluded.
