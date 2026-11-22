"""CAA 实验流程：扫描、对照、画图、保存。"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import numpy as np
import torch

from caa_core import (
    PROMPTS,
    build_direction,
    sentiment_logp_diff,
)

ALPHAS = [-8.0, -4.0, 0.0, 4.0, 8.0]
CONTROL_ALPHAS = (-8.0, 0.0, 8.0)
REAL_SEEDS = (0, 1, 2, 3, 4)
SHUFFLE_SEEDS = (0, 1, 2, 3, 4)


def _score_prompts(model, tokenizer, prompts, layer, alpha, vector, device) -> tuple[float, float, float]:
    vals = [
        sentiment_logp_diff(model, tokenizer, prompt, layer, alpha, vector, device)
        for prompt in prompts
    ]
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    win_rate = sum(v > 0 for v in vals) / len(vals)
    return mean, std, win_rate


def run_sweep(model, tokenizer, device, prompts, layers, alphas=ALPHAS) -> list[dict]:
    """扫描 layer × alpha，每个格点返回该组 prompt 的平均值。"""
    rows = []
    for layer in layers:
        vector = build_direction(model, tokenizer, layer, device)
        print(f"\n[layer {layer}] direction ready")

        for alpha in alphas:
            mean, std, win_rate = _score_prompts(
                model, tokenizer, prompts, layer, alpha, vector, device
            )
            rows.append(
                {
                    "layer": layer,
                    "alpha": alpha,
                    "mean_sent_delta": mean,
                    "std_sent_delta": std,
                    "win_rate": win_rate,
                    "n_prompts": len(prompts),
                }
            )
            print(
                f"  alpha={alpha:+.1f}  dlogP={mean:+.4f}  "
                f"std={std:.4f}  pos>neg={win_rate:.0%}"
            )
    return rows


def evaluate_prompts(model, tokenizer, prompts, layer, alpha, vector, device) -> dict:
    """在一组 prompt 上算平均 Δ、标准差、win rate。"""
    mean, std, win_rate = _score_prompts(
        model, tokenizer, prompts, layer, alpha, vector, device
    )
    return {"mean": mean, "std": std, "win_rate": win_rate}


def _run_seed_control(model, tokenizer, layer, device, seeds, direction_key: str) -> list[dict]:
    rows = []
    for seed in seeds:
        vector = build_direction(
            model, tokenizer, layer, device, **{direction_key: seed}
        )
        print(f"\n[{direction_key} {seed}] direction ready")

        for alpha in CONTROL_ALPHAS:
            mean, _, _ = _score_prompts(
                model, tokenizer, PROMPTS, layer, alpha, vector, device
            )
            rows.append(
                {"seed": seed, "layer": layer, "alpha": alpha, "mean_delta": mean}
            )
            print(f"  alpha={alpha:+.1f}  dlogP={mean:+.4f}")
    return rows


def run_real_seed_control(model, tokenizer, layer, device) -> list[dict]:
    """真实方向：有放回重采样 20 对句子。"""
    return _run_seed_control(model, tokenizer, layer, device, REAL_SEEDS, "bootstrap_seed")


def run_shuffle_control(model, tokenizer, layer, device) -> list[dict]:
    """shuffle-label：随机翻转每一对的正负标签。"""
    return _run_seed_control(model, tokenizer, layer, device, SHUFFLE_SEEDS, "shuffle_seed")


def summarize_control(rows: list[dict]) -> dict:
    """把多个 seed 汇总成每个 alpha 的均值和标准差。"""
    summary = {}
    for alpha in CONTROL_ALPHAS:
        vals = [r["mean_delta"] for r in rows if r["alpha"] == alpha]
        summary[f"{alpha:+.0f}"] = {
            "mean": statistics.fmean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        }
    return summary


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted({r["layer"] for r in rows})
    alphas = sorted({r["alpha"] for r in rows})
    baseline = {
        (r["layer"], r["alpha"]): r["mean_sent_delta"]
        for r in rows
        if r["alpha"] == 0.0
    }
    matrix = np.zeros((len(layers), len(alphas)))
    for r in rows:
        i = layers.index(r["layer"])
        j = alphas.index(r["alpha"])
        matrix[i, j] = r["mean_sent_delta"] - baseline[(r["layer"], 0.0)]

    limit = max(abs(matrix.min()), abs(matrix.max()), 1e-6)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(alphas)), [f"{a:+.0f}" for a in alphas])
    ax.set_yticks(range(len(layers)), [f"L{layer}" for layer in layers])
    ax.set_xlabel("alpha (unit direction)")
    ax.set_ylabel("steering layer")
    ax.set_title("Toy CAA on GPT-2: Δ log P('great') - log P('awful'), relative to α=0")

    for i in range(len(layers)):
        for j in range(len(alphas)):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_control_plot(
    real: dict,
    real_seed_summary: dict,
    random_control: dict,
    shuffle_summary: dict,
    layer: int,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["alpha = -8", "alpha = +8"]
    x = np.arange(len(labels))
    width = 0.2
    series = [
        ("real (full 20)", [real["-8"], real["+8"]], "#d62728", -1.5),
        ("real bootstrap (5 seeds)", [real_seed_summary["-8"]["mean"], real_seed_summary["+8"]["mean"]], "#ff7f0e", -0.5),
        ("shuffle-label (5 seeds)", [shuffle_summary["-8"]["mean"], shuffle_summary["+8"]["mean"]], "#7f7f7f", 0.5),
        ("random direction", [random_control["-8"], random_control["+8"]], "#1f77b4", 1.5),
    ]

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for label, values, color, offset in series:
        bars = ax.bar(x + offset * width, values, width, label=label, color=color)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_ylabel("mean Δ log P('great') - log P('awful')")
    ax.set_title(f"Real vs bootstrap vs shuffled-label vs random (layer {layer})")
    ax.set_xticks(x, labels)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
