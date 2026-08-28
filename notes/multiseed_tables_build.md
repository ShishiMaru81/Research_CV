# Multi-seed table build

Source: `D:/Research_Own/Research_CV/results/multiseed/transfer_cell_mean_std.csv`

## Headline (from `table_summary_stats_multiseed.csv`)

```
                        statistic     value
        baseline_cross_mean_3seed  0.445034
      baseline_cross_mean_std_avg  0.056994
strong_cross_mean_available_seeds  0.502145
             strong_cross_std_avg  0.062159
 mean_paired_aug_delta_cell_means  0.063486
         cells_positive_aug_delta 14.000000
              wilcoxon_p_18_cells  0.007690
    strong_cells_missing_seed2024 12.000000
```

## Strong-aug cells with fewer than 3 seeds

| train | test | model | n_seeds |
|-------|------|-------|--------|
| brri_rice_disease_pest | riceleafbd | efficientnet_b0 | 2 |
| brri_rice_disease_pest | riceleafbd | mobilenetv2_100 | 2 |
| brri_rice_disease_pest | riceleafbd | resnet50 | 2 |
| dhan_shomadhan | brri_rice_disease_pest | mobilenetv2_100 | 2 |
| dhan_shomadhan | riceleafbd | efficientnet_b0 | 2 |
| dhan_shomadhan | riceleafbd | mobilenetv2_100 | 2 |
| dhan_shomadhan | riceleafbd | resnet50 | 2 |
| riceleafbd | brri_rice_disease_pest | efficientnet_b0 | 2 |
| riceleafbd | brri_rice_disease_pest | mobilenetv2_100 | 2 |
| riceleafbd | brri_rice_disease_pest | resnet50 | 2 |
| riceleafbd | dhan_shomadhan | efficientnet_b0 | 2 |
| riceleafbd | dhan_shomadhan | resnet50 | 2 |

**12 / 18** strong-aug cells lack seed 2024 locally.
