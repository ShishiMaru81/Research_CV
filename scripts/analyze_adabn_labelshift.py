"""Analyze whether label-prior shift is associated with AdaBN performance changes.

The executed AdaBN experiment recalibrated batch-normalization statistics on the
complete target *training* split and evaluated on the target test split.  This
script deliberately mirrors that provenance.  It never reads image pixels and
does not require a GPU.

Analysis output is staged in ``week14_results/``.  The freeze step is
responsible for promoting the validated CSV into ``frozen_results_v2/``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "frozen_results" / "manifest.csv"
DEFAULT_OUTPUT = ROOT / "week14_results" / "adabn_labelshift.csv"
DEFAULT_EXPECTED_ROWS = 18
EPSILON = 1e-12

# Prefer a frozen, full-precision experiment table.  The publication table is
# intentionally last because it rounds metrics to three decimal places.
ADABN_CANDIDATES = (
    ROOT / "frozen_results_v2" / "adabn_results.csv",
    ROOT / "adabn_results.csv",
    ROOT / "week12_baselines_and_adabn" / "adabn" / "adabn_results.csv",
    ROOT / "week12_results" / "adabn" / "adabn_results.csv",
    # Rounded publication tables are a last-resort fallback only.
    ROOT / "frozen_results" / "table_adabn.csv",
    ROOT / "week12_results" / "adabn" / "table_adabn.csv",
)

KEY_COLUMNS = ["train_dataset", "test_dataset", "model", "classes", "seed"]
REQUIRED_ADABN_COLUMNS = set(KEY_COLUMNS) | {
    "baseline_macro_f1",
    "adabn_macro_f1",
    "delta_macro_f1",
    "n_samples",
    "n_adapt_images",
}
REQUIRED_MANIFEST_COLUMNS = {
    "image_path",
    "dataset",
    "mapped_class",
    "split",
    "is_duplicate",
}
METRICS = ("kl_sym", "tv_distance", "chi2")

MODEL_DISPLAY = {
    "mobilenetv2_100": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet50",
}


class AnalysisError(RuntimeError):
    """Raised when the saved experiment cannot support the requested analysis."""


def parse_shared_classes(value: object) -> list[str]:
    """Parse a comma- or pipe-separated class list without changing its order."""

    if pd.isna(value):
        raise AnalysisError("Encountered a missing shared-class specification.")
    classes = [part.strip() for part in re.split(r"[|,]", str(value)) if part.strip()]
    if not classes:
        raise AnalysisError(f"Could not parse any shared classes from {value!r}.")
    if len(classes) != len(set(classes)):
        raise AnalysisError(f"Shared classes contain duplicates: {value!r}.")
    return classes


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AnalysisError(f"{label} is missing required columns: {missing}")


def _nonduplicate_mask(series: pd.Series) -> pd.Series:
    """Return a strict mask for rows encoded as non-duplicates."""

    if pd.api.types.is_bool_dtype(series.dtype):
        return ~series
    normalized = series.astype(str).str.strip().str.lower()
    valid_false = {"false", "0", "no"}
    valid_true = {"true", "1", "yes"}
    unknown = sorted(set(normalized) - valid_false - valid_true)
    if unknown:
        raise AnalysisError(f"Unrecognized is_duplicate values: {unknown[:5]}")
    return normalized.isin(valid_false)


def _validate_finite_numeric(
    frame: pd.DataFrame, columns: Iterable[str], label: str
) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise AnalysisError(f"{label}.{column} is not numeric: {exc}") from exc
        if not np.isfinite(values).all():
            raise AnalysisError(f"{label}.{column} contains NaN or infinite values.")


def _resolve_adabn_path(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else ROOT / explicit
        if not path.is_file():
            raise FileNotFoundError(f"AdaBN table does not exist: {path}")
        return path
    for path in ADABN_CANDIDATES:
        if path.is_file():
            return path
    checked = "\n  - ".join(str(path) for path in ADABN_CANDIDATES)
    raise FileNotFoundError(f"No AdaBN result table found. Checked:\n  - {checked}")


def _load_inputs(
    adabn_path: Path, manifest_path: Path, expected_rows: int | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adabn = pd.read_csv(adabn_path)
    manifest = pd.read_csv(manifest_path)
    _require_columns(adabn, REQUIRED_ADABN_COLUMNS, str(adabn_path))
    _require_columns(manifest, REQUIRED_MANIFEST_COLUMNS, str(manifest_path))

    if expected_rows is not None and len(adabn) != expected_rows:
        raise AnalysisError(
            f"Expected {expected_rows} AdaBN rows, found {len(adabn)} in {adabn_path}."
        )
    duplicated = adabn.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = adabn.loc[duplicated, KEY_COLUMNS].head().to_dict("records")
        raise AnalysisError(f"AdaBN table has duplicate experiment keys: {examples}")

    numeric = [
        "baseline_macro_f1",
        "adabn_macro_f1",
        "delta_macro_f1",
        "n_samples",
        "n_adapt_images",
    ]
    if "n_bn_layers" in adabn.columns:
        numeric.append("n_bn_layers")
    _validate_finite_numeric(adabn, numeric, str(adabn_path))

    expected_delta = pd.to_numeric(adabn["adabn_macro_f1"]) - pd.to_numeric(
        adabn["baseline_macro_f1"]
    )
    actual_delta = pd.to_numeric(adabn["delta_macro_f1"])
    # Publication tables round each metric to three decimals.  Keep the strict
    # check for full-precision experiment tables, but allow the known rounding
    # envelope when a rounded table is the only available fallback.
    delta_tolerance = 0.0015 if adabn_path.name == "table_adabn.csv" else 1e-12
    if not np.allclose(actual_delta, expected_delta, rtol=0.0, atol=delta_tolerance):
        bad = np.flatnonzero(
            ~np.isclose(actual_delta, expected_delta, rtol=0.0, atol=delta_tolerance)
        )
        raise AnalysisError(
            "delta_macro_f1 is inconsistent with adabn_macro_f1 - "
            f"baseline_macro_f1 at rows {bad[:5].tolist()}."
        )

    manifest = manifest.loc[_nonduplicate_mask(manifest["is_duplicate"])].copy()
    if manifest.empty:
        raise AnalysisError("No non-duplicate manifest rows remain.")
    return adabn, manifest


def class_prior(rows: pd.DataFrame, classes: Sequence[str]) -> np.ndarray:
    """Compute a class prior aligned to ``classes`` in float64."""

    if rows.empty:
        raise AnalysisError("Cannot compute a class prior from zero rows.")
    counts = rows["mapped_class"].value_counts().reindex(classes, fill_value=0)
    # Zero components are valid prior-shift observations.  Divergence metrics
    # apply epsilon smoothing for KL; rejecting them here would hide the most
    # extreme (and scientifically relevant) shift.
    prior = counts.to_numpy(dtype=np.float64)
    prior /= prior.sum()
    return prior


def divergence_metrics(
    prior_src: Sequence[float], prior_tgt: Sequence[float], eps: float = EPSILON
) -> tuple[float, float, float]:
    """Return symmetric KL, total variation, and directional Pearson chi-square.

    KL is evaluated after epsilon clipping and renormalization.  The chi-square
    expression follows the task specification and is directional because its
    denominator is the source prior.
    """

    src = np.asarray(prior_src, dtype=np.float64)
    tgt = np.asarray(prior_tgt, dtype=np.float64)
    if src.shape != tgt.shape or src.ndim != 1 or src.size == 0:
        raise AnalysisError("Prior vectors must be nonempty one-dimensional peers.")
    if not np.isfinite(src).all() or not np.isfinite(tgt).all():
        raise AnalysisError("Prior vectors contain NaN or infinite values.")
    if (src < 0).any() or (tgt < 0).any():
        raise AnalysisError("Prior vectors cannot contain negative values.")
    if not np.isclose(src.sum(), 1.0) or not np.isclose(tgt.sum(), 1.0):
        raise AnalysisError("Prior vectors must each sum to one.")
    if not np.isfinite(eps) or eps <= 0:
        raise AnalysisError("epsilon must be positive.")

    src_kl = np.clip(src, eps, None)
    tgt_kl = np.clip(tgt, eps, None)
    src_kl /= src_kl.sum()
    tgt_kl /= tgt_kl.sum()
    kl_src_tgt = float(np.sum(src_kl * np.log(src_kl / tgt_kl)))
    kl_tgt_src = float(np.sum(tgt_kl * np.log(tgt_kl / src_kl)))
    kl_sym = 0.5 * (kl_src_tgt + kl_tgt_src)
    tv_distance = 0.5 * float(np.abs(src - tgt).sum())
    chi2 = float(np.sum(np.square(src - tgt) / (src + eps)))
    return kl_sym, tv_distance, chi2


def _recorded_bn_counts(adabn: pd.DataFrame) -> dict[str, int] | None:
    if "n_bn_layers" not in adabn.columns:
        return None
    counts: dict[str, int] = {}
    for model_name, group in adabn.groupby("model", sort=True):
        numeric = pd.to_numeric(group["n_bn_layers"], errors="raise")
        if not np.equal(numeric, np.floor(numeric)).all() or (numeric <= 0).any():
            raise AnalysisError(f"Invalid recorded BN-layer count for {model_name}.")
        unique = sorted({int(value) for value in numeric})
        if len(unique) != 1:
            raise AnalysisError(
                f"Recorded BN-layer count is inconsistent for {model_name}: {unique}"
            )
        counts[str(model_name)] = unique[0]
    return counts


def _instantiate_bn_counts(model_names: Iterable[str]) -> dict[str, int]:
    """Fallback for tables that predate saved BN counts (imports are intentionally lazy)."""

    try:
        import torch.nn as nn
        import timm
    except ImportError as exc:
        raise AnalysisError(
            "The AdaBN table has no n_bn_layers column and torch/timm are unavailable. "
            "Use the full-precision adabn_results.csv or install the project dependencies."
        ) from exc

    counts: dict[str, int] = {}
    for model_name in sorted(set(model_names)):
        # Pretrained weights cannot affect architecture and would cause a network download.
        model = timm.create_model(model_name, num_classes=2, pretrained=False)
        count = sum(
            1 for module in model.modules() if isinstance(module, nn.BatchNorm2d)
        )
        if count <= 0:
            raise AnalysisError(f"Model {model_name!r} contains no BatchNorm2d layers.")
        counts[model_name] = count
    return counts


def resolve_bn_counts(adabn: pd.DataFrame) -> tuple[dict[str, int], str]:
    recorded = _recorded_bn_counts(adabn)
    if recorded is not None:
        return recorded, "recorded by the executed AdaBN run"
    return (
        _instantiate_bn_counts(adabn["model"].astype(str)),
        "instantiated with pretrained=False",
    )


def analyze_rows(
    adabn: pd.DataFrame,
    manifest: pd.DataFrame,
    bn_counts: dict[str, int],
    *,
    adapt_split: str = "train",
    eps: float = EPSILON,
) -> pd.DataFrame:
    """Compute one divergence record per saved AdaBN result row."""

    records: list[dict[str, object]] = []
    for row in adabn.itertuples(index=False):
        classes = parse_shared_classes(row.classes)
        source_rows = manifest.loc[
            (manifest["dataset"] == row.train_dataset)
            & (manifest["split"] == "train")
            & manifest["mapped_class"].isin(classes)
        ]
        target_rows = manifest.loc[
            (manifest["dataset"] == row.test_dataset)
            & (manifest["split"] == adapt_split)
            & manifest["mapped_class"].isin(classes)
        ]

        recorded_n = int(row.n_adapt_images)
        if len(target_rows) != recorded_n:
            raise AnalysisError(
                f"{row.train_dataset}->{row.test_dataset} ({row.classes}) has "
                f"{len(target_rows)} target-{adapt_split} rows but the AdaBN run records "
                f"n_adapt_images={recorded_n}. A manifest-order prefix would not "
                "reconstruct the shuffled adaptation loader, so analysis is aborted."
            )
        prior_src = class_prior(source_rows, classes)
        prior_tgt = class_prior(target_rows, classes)
        kl_sym, tv_distance, chi2 = divergence_metrics(prior_src, prior_tgt, eps)

        model_name = str(row.model)
        if model_name not in bn_counts:
            raise AnalysisError(f"No BN-layer count is available for {model_name!r}.")
        records.append(
            {
                "train_dataset": row.train_dataset,
                "test_dataset": row.test_dataset,
                "model": model_name,
                "classes": "|".join(classes),
                "seed": int(row.seed),
                "delta_macro_f1": float(row.delta_macro_f1),
                "kl_sym": kl_sym,
                "tv_distance": tv_distance,
                "chi2": chi2,
                "bn_layers": int(bn_counts[model_name]),
                "source_split": "train",
                "target_adapt_split": adapt_split,
                "n_source_train": int(len(source_rows)),
                "n_target_adapt": int(len(target_rows)),
                "prior_src": json.dumps(
                    dict(zip(classes, map(float, prior_src))), separators=(",", ":")
                ),
                "prior_tgt": json.dumps(
                    dict(zip(classes, map(float, prior_tgt))), separators=(",", ":")
                ),
            }
        )

    result = pd.DataFrame.from_records(records)
    result = result.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    if result.duplicated(KEY_COLUMNS).any():
        raise AnalysisError("Analysis unexpectedly produced duplicate experiment keys.")
    return result


def spearman_record(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    values = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(values)
    if n < 2 or values["x"].nunique() < 2 or values["y"].nunique() < 2:
        return {"rho": float("nan"), "p_value": float("nan"), "n": n}
    rho, p_value = spearmanr(values["x"], values["y"], nan_policy="omit")
    return {"rho": float(rho), "p_value": float(p_value), "n": n}


def exact_permutation_p(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Exact two-sided permutation p-value for a six-pair Spearman sensitivity check."""

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if len(x_array) > 8:
        raise AnalysisError("Exact permutation sensitivity is capped at eight pairs.")
    observed = float(spearmanr(x_array, y_array).statistic)
    if not np.isfinite(observed):
        return observed, float("nan")
    extreme = 0
    total = 0
    for permutation in itertools.permutations(y_array.tolist()):
        statistic = float(spearmanr(x_array, permutation).statistic)
        total += 1
        if abs(statistic) >= abs(observed) - 1e-12:
            extreme += 1
    return observed, extreme / total


def correlation_tables(
    result: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float | int],
    dict[str, float | int],
]:
    overall = pd.DataFrame(
        [
            {
                "metric": metric,
                **spearman_record(result[metric], result["delta_macro_f1"]),
            }
            for metric in METRICS
        ]
    )

    within_rows: list[dict[str, object]] = []
    for model_name, group in result.groupby("model", sort=True):
        for metric in METRICS:
            within_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    **spearman_record(group[metric], group["delta_macro_f1"]),
                }
            )
    within = pd.DataFrame(within_rows)

    pair_keys = ["train_dataset", "test_dataset", "classes"]
    pair_rows: list[dict[str, object]] = []
    pair_groups = result.groupby(pair_keys, sort=True)
    expected_models = set(result["model"].astype(str))
    for pair, group in pair_groups:
        if set(group["model"].astype(str)) != expected_models:
            raise AnalysisError(
                f"Pair {pair} does not contain the same model set as the other pairs."
            )
        for metric in METRICS:
            if group[metric].nunique(dropna=False) != 1:
                raise AnalysisError(
                    f"{metric} is not invariant across models for pair {pair}."
                )

    pair_level = pair_groups.agg(
        delta_macro_f1=("delta_macro_f1", "mean"),
        kl_sym=("kl_sym", "first"),
        tv_distance=("tv_distance", "first"),
        chi2=("chi2", "first"),
    ).reset_index()
    for metric in METRICS:
        rho, exact_p = exact_permutation_p(
            pair_level[metric], pair_level["delta_macro_f1"]
        )
        pair_rows.append(
            {
                "metric": metric,
                "rho": rho,
                "exact_p_value": exact_p,
                "bonferroni_p_value": min(1.0, exact_p * len(METRICS)),
                "n_pairs": len(pair_level),
            }
        )
    pair_sensitivity = pd.DataFrame(pair_rows)

    depth_magnitude = spearman_record(
        result["bn_layers"], result["delta_macro_f1"].abs()
    )
    depth_harm = spearman_record(result["bn_layers"], -result["delta_macro_f1"])
    return overall, within, pair_sensitivity, depth_magnitude, depth_harm


def _format_float(value: object) -> str:
    numeric = float(value)
    return "nan" if not np.isfinite(numeric) else f"{numeric:+.4f}"


def print_report(result: pd.DataFrame) -> None:
    overall, within, pair_sensitivity, depth_magnitude, depth_harm = correlation_tables(
        result
    )
    negative = int((result["delta_macro_f1"] < 0).sum())
    positive = int((result["delta_macro_f1"] > 0).sum())
    zero = int((result["delta_macro_f1"] == 0).sum())

    print(
        f"\nADA BN OUTCOME COUNTS: negative={negative}, positive={positive}, zero={zero}"
    )
    print("The premise 'uniformly hurt' is true only if every delta is negative.")

    print(f"\nOVERALL CORRELATIONS (Spearman; {len(result)} model-pair rows):")
    display = overall.copy()
    display["rho"] = display["rho"].map(_format_float)
    display["p_value"] = display["p_value"].map(lambda x: f"{float(x):.6f}")
    print(display.to_string(index=False))

    print("\nWITHIN-MODEL CORRELATIONS:")
    display = within.copy()
    display["model"] = display["model"].map(
        lambda value: MODEL_DISPLAY.get(value, value)
    )
    display["rho"] = display["rho"].map(_format_float)
    display["p_value"] = display["p_value"].map(lambda x: f"{float(x):.6f}")
    print(display.to_string(index=False))

    print(
        "\nPAIR-LEVEL SENSITIVITY (mean delta over models; exact two-sided permutation p):"
    )
    display = pair_sensitivity.copy()
    display["rho"] = display["rho"].map(_format_float)
    display["exact_p_value"] = display["exact_p_value"].map(lambda x: f"{float(x):.6f}")
    display["bonferroni_p_value"] = display["bonferroni_p_value"].map(
        lambda x: f"{float(x):.6f}"
    )
    print(display.to_string(index=False))

    print("\nBN-LAYER ANALYSIS (descriptive; only three BN-count levels):")
    print(
        "Spearman(bn_layers, |delta_macro_f1|): "
        f"rho={_format_float(depth_magnitude['rho'])}, "
        f"p={float(depth_magnitude['p_value']):.6f}, n={depth_magnitude['n']}"
    )
    print(
        "Spearman(bn_layers, harm=-delta_macro_f1): "
        f"rho={_format_float(depth_harm['rho'])}, "
        f"p={float(depth_harm['p_value']):.6f}, n={depth_harm['n']}"
    )
    model_summary = (
        result.assign(
            harm=-result["delta_macro_f1"], magnitude=result["delta_macro_f1"].abs()
        )
        .groupby("model", as_index=False)
        .agg(
            bn_layers=("bn_layers", "first"),
            mean_delta=("delta_macro_f1", "mean"),
            mean_harm=("harm", "mean"),
            mean_magnitude=("magnitude", "mean"),
            n_negative=("delta_macro_f1", lambda values: int((values < 0).sum())),
            n=("delta_macro_f1", "size"),
        )
        .sort_values("bn_layers")
    )
    print(model_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    pair_directional = pair_sensitivity.loc[
        (pair_sensitivity["exact_p_value"] * len(METRICS) < 0.05)
        & (pair_sensitivity["rho"] < 0)
    ]
    pair_opposite = pair_sensitivity.loc[
        (pair_sensitivity["exact_p_value"] * len(METRICS) < 0.05)
        & (pair_sensitivity["rho"] > 0)
    ]
    print("\nINTERPRETATION (observational, not causal):")
    print(
        "The 18-row Spearman p-values are nominal/descriptive only because each "
        "domain pair is repeated across models. Primary sensitivity inference uses "
        "six pair means with an exact permutation p-value and Bonferroni adjustment "
        "across the three divergence metrics."
    )
    if not pair_directional.empty:
        print(
            "The pair-level sensitivity has a significant association in the predicted "
            "direction after the three-metric adjustment."
        )
    elif not pair_opposite.empty:
        print(
            "The pair-level sensitivity has a significant association, but its positive "
            "direction is opposite to the proposed harm mechanism."
        )
    else:
        print(
            "No divergence metric has p < 0.05 in the predicted negative direction; "
            "the label-prior mechanism is unresolved by this analysis."
        )
    print(
        "Caveat: the overall 18 rows repeat six ordered dataset pairs across three "
        "models, symmetric KL/TV have additional reciprocal-pair ties, and BN-layer "
        "count is not a validated measure of architectural depth. Treat p-values as "
        "exploratory. AdaBN running statistics were also updated in shuffled batches, "
        "so global class priors are only a proxy for the statistics actually accumulated."
    )

    print("\nFULL ANALYSIS CSV:")
    print(result.to_csv(index=False).rstrip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adabn-table",
        type=Path,
        default=None,
        help="Full-precision AdaBN CSV (auto-detected by default).",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--adapt-split",
        choices=("train", "test"),
        default="train",
        help="Split seen by AdaBN; the executed experiment used target train.",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=DEFAULT_EXPECTED_ROWS,
        help="Fail if the AdaBN table does not have this many rows; use 0 to disable.",
    )
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adabn_path = _resolve_adabn_path(args.adabn_table)
    manifest_path = (
        args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    )
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    expected_rows = None if args.expected_rows == 0 else args.expected_rows

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    adabn, manifest = _load_inputs(adabn_path, manifest_path, expected_rows)
    print(f"Loaded AdaBN table: {adabn_path} ({len(adabn)} rows)")
    print(f"Loaded manifest: {manifest_path} ({len(manifest)} non-duplicate rows)")
    if adabn_path.name == "table_adabn.csv":
        print("WARNING: table_adabn.csv may contain rounded publication values.")

    bn_counts, bn_source = resolve_bn_counts(adabn)
    print(f"BN-layer counts: {bn_counts} ({bn_source})")
    print(f"Target adaptation split: {args.adapt_split} (must match n_adapt_images)")

    result = analyze_rows(
        adabn,
        manifest,
        bn_counts,
        adapt_split=args.adapt_split,
        eps=args.epsilon,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, lineterminator="\n")
    print(f"Saved: {output_path} ({len(result)} rows)")
    print_report(result)


if __name__ == "__main__":
    main()
