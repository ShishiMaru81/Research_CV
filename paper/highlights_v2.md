# Highlights (v2)

- Cross-dataset macro-F1 0.45 versus in-dataset 0.72 on Bangladeshi rice collections.
- White-background F1 0.85 exceeds field 0.71 and cross-field 0.57 on fixed Dhan model.
- Strong augmentation adds +0.06 cross-F1 (14/18; Wilcoxon p~0.00769); LODO wins 3/9.
- AdaBN mean delta -0.06; ResNet50 never improves under BatchNorm recalibration alone.
- Geometric transforms recover 67% of ResNet50 strong-aug cross-F1 gain on six pairs.
