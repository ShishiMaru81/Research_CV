# Dataset Inventory (Week 1)

## Source-to-target organization
- data/raw/riceleafbd/: populated from Dataset/Original Images/Original Images
- data/raw/dhan_shomadhan/: populated from Dataset/znsxdctwtt-1/Dhan-Shomadhan
- data/raw/brri_rice_disease_pest/: extracted from Dataset/Rice Leaf Disease and Pest Dataset Overview/Rice Leaf Disease and Pest Dataset Overview/Rice Dataset.zip using only Rice Dataset/Original Dataset/

## Image counts by folder label
### riceleafbd (total: 1560)
- Bacterial Leaf Blight: 422
- Brown Spot: 356
- Healthy Leaf: 252
- Tungro Virus: 530

### dhan_shomadhan (total: 1106)
- Brown Spot: 90
- Browon Spot: 49
- Leaf Scaled: 217
- Rice Blast: 272
- Rice Tungro: 119
- Rice Turgro: 76
- Shath Blight: 219
- Sheath Blight: 64

### brri_rice_disease_pest (total on disk: 2769; used in study: 2753)
- Healthy: 603
- Insect: 281
- Leaf Scald: 421
- Rice: 16 (**excluded** — out-of-scope / ambiguous; see `notes/label_mapping.md`)
- Rice Blast: 696
- Rice Leaffolder: 247
- Rice Stripes: 266
- Rice Tungro: 239

## Noted spelling/label variants
- Browon Spot -> variant of brown_spot
- Rice Turgro -> variant of tungro
- Shath Blight -> likely variant of sheath_blight
- Rice Leaffolder -> variant of leaf_folder
- Healthy Leaf -> variant of healthy

## Augmentation and background observations
- BRRI archive contains both Original Dataset and Augmented Dataset; only originals were extracted.
- dhan_shomadhan includes Field Background and White Background sub-structure.
