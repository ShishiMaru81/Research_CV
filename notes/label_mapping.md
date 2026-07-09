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

## Flagged class needing verification in Week 2
- Rice (16 images in BRRI originals) does not directly map from canonical list and must be resolved before manifest build.
