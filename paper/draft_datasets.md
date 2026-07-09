# Datasets (draft for paper)

We evaluate cross-dataset generalization across three Bangladeshi rice leaf disease collections: RiceLeafBD, Dhan-Shomadhan, and the BRRI Rice Leaf Disease and Pest dataset. After excluding pre-augmented images and one out-of-scope BRRI folder (`Rice`, 16 general/panicle images), the harmonized corpus contains 5,419 original images.

Raw class names were mapped to a shared canonical taxonomy (healthy, bacterial leaf blight, brown spot, tungro, rice blast, scald, sheath blight, leaf folder, insect, stripes), including documented spelling variants (e.g., Browon Spot, Rice Turgro, Shath Blight). Stratified 70/15/15 train/val/test splits were frozen per dataset and class with a fixed seed. Cross-dataset near-duplicate search with perceptual hashing (threshold 2) found no true shared copies.

Shared-class transfer pairs used in experiments are: RiceLeafBD ∩ Dhan-Shomadhan = {brown_spot, tungro}; Dhan-Shomadhan ∩ BRRI = {rice_blast, scald, tungro}; RiceLeafBD ∩ BRRI = {healthy, tungro}. Dhan-Shomadhan uniquely provides both field and white-background images, enabling a later background-confound diagnosis.
