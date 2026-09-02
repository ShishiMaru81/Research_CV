"""Profile deployment characteristics of CNN backbones (Week 10, CPU)."""

from __future__ import annotations

import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "frozen_results_v2" / "deployment_profile.csv"
OUT_JSON = ROOT / "frozen_results_v2" / "deployment_env.json"

MODELS = ("mobilenetv2_100", "efficientnet_b0", "resnet50")
NUM_CLASSES = 2
IMAGE_SIZE = 224
WARMUP_RUNS = 10
TIMED_RUNS = 50
SEED = 42


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"torch.manual_seed({seed})")
    print(f"numpy.random.seed({seed})")


def _count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _measure_gflops(model: torch.nn.Module, device: torch.device) -> float:
    """Return GFLOPs at batch 1, 224x224. Requires fvcore or ptflops."""
    model.eval()
    example = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    fvcore_error: ImportError | None = None
    try:
        from fvcore.nn import FlopCountAnalysis

        flops = FlopCountAnalysis(model, example).total()
        if flops is None or flops <= 0:
            raise RuntimeError(f"fvcore returned invalid flop count: {flops}")
        return flops / 1e9
    except ImportError as exc:
        fvcore_error = exc

    try:
        from ptflops import get_model_complexity_info

        # ptflops expects CPU model for counting
        model_cpu = model.to("cpu")
        example_cpu = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        macs, _ = get_model_complexity_info(
            model_cpu,
            (3, IMAGE_SIZE, IMAGE_SIZE),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        model.to(device)
        if macs is None or macs <= 0:
            raise RuntimeError(f"ptflops returned invalid MAC count: {macs}")
        # 1 MAC ≈ 2 FLOPs; report GFLOPs as FLOPs/1e9
        return (2.0 * macs) / 1e9
    except ImportError as ptflops_error:
        raise ImportError(
            "GFLOPs measurement requires fvcore or ptflops. "
            "Install with: pip install fvcore  OR  pip install ptflops"
        ) from (ptflops_error if fvcore_error is None else fvcore_error)


def _measure_latency_ms(
    model: torch.nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
    example = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            _ = model(example)
            if device.type == "cuda":
                torch.cuda.synchronize()

        times_ms: list[float] = []
        for _ in range(TIMED_RUNS):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(example)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _model_size_mb(model: torch.nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / (1024.0 * 1024.0)


def main() -> None:
    _set_seeds(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print("Latency measured on CUDA (with synchronize).")
    else:
        device = torch.device("cpu")
        print("CUDA not available — latency measured on CPU.")

    rows: list[dict[str, float | str]] = []
    for model_name in MODELS:
        print(f"\nProfiling {model_name}...")
        model = timm.create_model(
            model_name, pretrained=False, num_classes=NUM_CLASSES
        )
        model.to(device)
        model.eval()

        n_params, n_trainable = _count_params(model)
        gflops = _measure_gflops(model, device)
        lat_mean, lat_std = _measure_latency_ms(model, device)
        size_mb = _model_size_mb(model)

        rows.append(
            {
                "model": model_name,
                "n_params": n_params,
                "n_trainable": n_trainable,
                "gflops": gflops,
                "cpu_latency_ms_mean": lat_mean,
                "cpu_latency_ms_std": lat_std,
                "model_size_mb": size_mb,
            }
        )
        print(
            f"  params={n_params:,} gflops={gflops:.4f} "
            f"latency={lat_mean:.2f}±{lat_std:.2f} ms size={size_mb:.2f} MB"
        )

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    env = {
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "n_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "device_used_for_latency": str(device),
        "seed": SEED,
    }
    OUT_JSON.write_text(json.dumps(env, indent=2), encoding="utf-8")

    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    print(f"\nShape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()
