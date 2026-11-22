"""测量 CAA steering 对 GPT-2 语言建模能力的影响。

做法：在 WikiText-2 的固定文本块上算 next-token cross-entropy。
对比 baseline、真实情感方向、随机方向、shuffle-label 方向。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from caa_core import add_vector_hook, build_direction, load_model, pick_device

BLOCK_SIZE = 512
MAX_BLOCKS = 8
BATCH_SIZE = 2
DEFAULT_DATA = Path(__file__).parent / "data" / "wikitext2_test_256lines.txt"
RESULTS_DIR = Path(__file__).parent / "results"


def load_blocks(tokenizer, path: Path, block_size: int, max_blocks: int) -> torch.Tensor:
    """把纯文本切成 [n_blocks, block_size] 的 token 块。"""
    text = path.read_text(encoding="utf-8")
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    need = block_size * max_blocks
    if ids.numel() < need:
        raise ValueError(f"text too short: {ids.numel()} tokens < {need}")
    ids = ids[:need]
    return ids.view(max_blocks, block_size)


def evaluate_loss(model, blocks: torch.Tensor, layer: int, alpha: float, vector) -> float:
    """固定文本块上的平均 next-token loss；vector=None 表示不干预。"""
    handle = None
    if vector is not None:
        # start=0：所有 token 位置都加，测的是持续干预下的语言建模能力。
        handle = model.transformer.h[layer].register_forward_hook(
            add_vector_hook(vector, alpha, start=0)
        )

    total_loss = 0.0
    total_tokens = 0
    try:
        with torch.no_grad():
            for start in range(0, blocks.shape[0], BATCH_SIZE):
                batch = blocks[start : start + BATCH_SIZE].to(model.device)
                out = model(input_ids=batch, labels=batch)
                n = batch.numel()
                total_loss += out.loss.item() * n
                total_tokens += n
    finally:
        if handle is not None:
            handle.remove()

    return total_loss / total_tokens


def random_direction(model, seed: int = 0) -> torch.Tensor:
    """和情感无关的随机单位方向，作为对照。"""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(model.config.n_embd, generator=g)
    return (v / v.norm()).to(model.device)


def run_layer(model, tokenizer, blocks, layer: int, device: str) -> list[dict]:
    real = build_direction(model, tokenizer, layer, device)
    shuffled = build_direction(model, tokenizer, layer, device, shuffle_seed=0)
    random_v = random_direction(model)

    conditions = [
        ("baseline", 0.0, None),
        ("real_-8", -8.0, real),
        ("real_+8", 8.0, real),
        ("random_-8", -8.0, random_v),
        ("random_+8", 8.0, random_v),
        ("shuffle_+8", 8.0, shuffled),
    ]

    rows = []
    baseline_loss = None
    for name, alpha, vector in conditions:
        loss = evaluate_loss(model, blocks, layer, alpha, vector)
        if name == "baseline":
            baseline_loss = loss
        ppl = torch.exp(torch.tensor(loss)).item()
        delta = loss - baseline_loss
        rows.append(
            {
                "layer": layer,
                "condition": name,
                "alpha": alpha,
                "loss": round(loss, 6),
                "ppl": round(ppl, 4),
                "delta_loss": round(delta, 6),
            }
        )
        print(f"L{layer:<2} {name:<12} loss={loss:.4f}  ppl={ppl:.2f}  delta={delta:+.4f}")

    base_ppl = torch.exp(torch.tensor(baseline_loss)).item()
    for row in rows:
        row["delta_ppl_pct"] = round((row["ppl"] / base_ppl - 1.0) * 100.0, 3)
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    fields = ["layer", "condition", "alpha", "loss", "ppl", "delta_loss", "delta_ppl_pct"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict], path: Path, n_blocks: int) -> None:
    layers = sorted({r["layer"] for r in rows})
    conditions = [
        "baseline",
        "real_-8",
        "real_+8",
        "random_-8",
        "random_+8",
        "shuffle_+8",
    ]
    colors = ["#9e9e9e", "#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 5), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        values = []
        for name in conditions:
            row = next(r for r in rows if r["layer"] == layer and r["condition"] == name)
            values.append(row["delta_ppl_pct"])
        bars = ax.bar(range(len(conditions)), values, color=colors, alpha=0.85)
        ax.set_title(f"Layer {layer}")
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=35, ha="right")
        ax.set_ylabel("perplexity increase vs baseline (%)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:+.2f}%",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )

    fig.suptitle(
        f"GPT-2 perplexity change under CAA steering ({n_blocks} blocks x 512 tokens)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--blocks", type=int, default=MAX_BLOCKS)
    parser.add_argument("--layers", type=str, default="7,8")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    device = pick_device()
    print(f"device: {device}")
    model, tokenizer = load_model(device)
    blocks = load_blocks(tokenizer, args.data, BLOCK_SIZE, args.blocks)
    print(f"data: {args.data}  blocks: {tuple(blocks.shape)}")

    rows = []
    for layer in layers:
        rows.extend(run_layer(model, tokenizer, blocks, layer, device))

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "capability_loss.csv"
    png_path = RESULTS_DIR / "capability_loss.png"
    json_path = RESULTS_DIR / "capability_loss.json"
    save_csv(rows, csv_path)
    save_plot(rows, png_path, blocks.shape[0])
    json_path.write_text(
        json.dumps(
            {
                "data": str(args.data),
                "block_size": BLOCK_SIZE,
                "n_blocks": blocks.shape[0],
                "layers": layers,
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote: {csv_path}")
    print(f"wrote: {png_path}")
    print(f"wrote: {json_path}")


if __name__ == "__main__":
    main()
