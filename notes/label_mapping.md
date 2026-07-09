# Label Mapping Specification (Week 1)

Canonical labels:
- healthy
- bacterial_leaf_blight
- brown_spot
- tungro
- rice_blast
- scald
- sheath_blight
- leaf_folder
- insect
- stripes

## Raw -> canonical mapping (case-insensitive)
- Healthy, Healthy Rice Leaf, Healthy Leaf -> healthy
- Bacterial Leaf Blight, BLB -> bacterial_leaf_blight
- Brown Spot, Browon Spot -> brown_spot
- Tungro, Tungro Virus, Tungro Disease, Rice Turngo, Rice Tungro, Rice Turgro -> tungro
- Rice Blast, Leaf Blast, Blast -> rice_blast
- Leaf Scald, Leaf Scaled, Scald -> scald
- Sheath Blight, Steath Blight, Stealth Blight, Shath Blight -> sheath_blight
- Leaf-folder, Leaffolder, Leaf Folder, Leaf-folder Injury, Rice Leaffolder -> leaf_folder
- Insect Infestation, Insect Damage, Insect -> insect
- Rice Stripes, Leaf Stripes -> stripes

## Shared class sets for transfer experiments
- riceleafbd ∩ dhan_shomadhan = {brown_spot, tungro}
- dhan_shomadhan ∩ brri_rice_disease_pest = {rice_blast, scald, tungro}
- riceleafbd ∩ brri_rice_disease_pest = {healthy, tungro}
- all three = {tungro}

## Resolved Week 2 edge case: BRRI Rice
- Decision: **exclude** (do not map to any canonical class).
- Count: 16 original images under `brri_rice_disease_pest/Rice/`.
- Rationale:
  - Mendeley description lists Rice as "General rice leaf images", not a disease/pest class.
  - Peer-reviewed Data in Brief paper reports 7 classes and omits Rice from the official class set.
  - Spot-check of samples shows panicles with grain discoloration / mixed content, not leaf-disease classes used in this study.
- Implementation: `EXCLUDED_RAW_CLASSES` in `src/build_manifest.py` skips this folder and reports the exclusion count.
