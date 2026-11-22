"""玩具 CAA 的命令行入口：跑扫描、对照、held-out，保存结果。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from caa_core import (
    HELDOUT_PROMPTS,
    PAIRS,
    PROMPTS,
    TRAIN_PROMPTS,
    build_direction,
    load_model,
    pick_device,
    sentiment_logp_diff,
    top_next_tokens,
)
from caa_experiments import (
    ALPHAS,
    REAL_SEEDS,
    SHUFFLE_SEEDS,
    evaluate_prompts,
    run_real_seed_control,
    run_shuffle_control,
    run_sweep,
    save_control_plot,
    save_csv,
    save_heatmap,
    summarize_control,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LAYERS = [7, 8, 9]
TAG_BY_LAYERS = {(4, 6, 8, 10): "coarse", (7, 8, 9): "fine"}


def parse_layers(value: str) -> list[int]:
    """解析 --layers 7,8,9，并检查 GPT-2 small 的层号范围。"""
    try:
        layers = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers 需要逗号分隔的整数，例如 7,8,9") from exc
    if not layers:
        raise argparse.ArgumentTypeError("至少给一层，例如 7,8,9")
    if any(layer < 0 or layer >= 12 for layer in layers):
        raise argparse.ArgumentTypeError("GPT-2 small 只有 0–11 层")
    return layers


def parse_tag(value: str) -> str:
    """输出文件后缀，只允许字母、数字、下划线、连字符。"""
    tag = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise argparse.ArgumentTypeError("tag 只能包含字母、数字、下划线、连字符")
    return tag


def best_positive(rows: list[dict]) -> dict:
    return max((r for r in rows if r["alpha"] > 0), key=lambda r: r["mean_sent_delta"])


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true", help="allow Hugging Face Hub online check")
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default=None,
        help="comma-separated layer ids, e.g. 4,6,8,10 or 7,8,9; default: LAYERS",
    )
    parser.add_argument(
        "--tag",
        type=parse_tag,
        default=None,
        help="suffix for output files; empty = auto (coarse/fine/run)",
    )
    args = parser.parse_args()

    layers = LAYERS if args.layers is None else args.layers
    tag = args.tag or TAG_BY_LAYERS.get(tuple(layers), "run")
    suffix = f"_{tag}"
    print(f"layers: {layers}")
    print(f"output suffix: {suffix}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    model, tokenizer = load_model(device)
    print(f"prompt tokens: {len(tokenizer(PROMPTS[0], add_special_tokens=False).input_ids)}")
    print(f"prefix tokens: {len(tokenizer(' really', add_special_tokens=False).input_ids)}")

    # 全量 20 个 prompt：描述性主表。
    rows = run_sweep(model, tokenizer, device, PROMPTS, layers)
    save_csv(rows, RESULTS / f"layer_alpha{suffix}.csv")
    save_heatmap(rows, RESULTS / f"layer_alpha{suffix}.png")

    # 只用 15 个训练 prompt 选层和 alpha。
    train_rows = run_sweep(model, tokenizer, device, TRAIN_PROMPTS, layers)
    save_csv(train_rows, RESULTS / f"layer_alpha_train{suffix}.csv")
    save_heatmap(train_rows, RESULTS / f"layer_alpha_train{suffix}.png")

    full_best = best_positive(rows)
    train_best = best_positive(train_rows)
    best_layer = int(train_best["layer"])
    print(
        f"\nfull best (descriptive): layer={full_best['layer']}, "
        f"alpha={full_best['alpha']:+.1f}, dlogP={full_best['mean_sent_delta']:+.4f}"
    )
    print(
        f"train best (used for held-out): layer={best_layer}, alpha={train_best['alpha']:+.1f}, "
        f"dlogP={train_best['mean_sent_delta']:+.4f}"
    )

    best_vector = build_direction(model, tokenizer, best_layer, device)

    # 随机方向对照。
    torch.manual_seed(0)
    random_vector = torch.randn_like(best_vector)
    random_vector = random_vector / random_vector.norm().clamp_min(1e-8)

    random_control = {}
    for alpha in (-8.0, 8.0):
        vals = [
            sentiment_logp_diff(model, tokenizer, prompt, best_layer, alpha, random_vector, device)
            for prompt in PROMPTS
        ]
        random_control[f"{alpha:+.0f}"] = sum(vals) / len(vals)
    print(f"\nrandom control at layer {best_layer}: {random_control}")

    real_control = {
        f"{alpha:+.0f}": next(
            r["mean_sent_delta"]
            for r in rows
            if r["layer"] == best_layer and r["alpha"] == alpha
        )
        for alpha in (-8.0, 8.0)
    }

    real_seed_rows = run_real_seed_control(model, tokenizer, best_layer, device)
    save_csv(real_seed_rows, RESULTS / f"real_seed_control{suffix}.csv")
    real_seed_summary = summarize_control(real_seed_rows)
    print(f"\nreal bootstrap at layer {best_layer}: {real_seed_summary}")

    shuffle_rows = run_shuffle_control(model, tokenizer, best_layer, device)
    save_csv(shuffle_rows, RESULTS / f"shuffle_control{suffix}.csv")
    shuffle_summary = summarize_control(shuffle_rows)
    print(f"\nshuffle-label control at layer {best_layer}: {shuffle_summary}")

    save_control_plot(
        real_control,
        real_seed_summary,
        random_control,
        shuffle_summary,
        best_layer,
        RESULTS / f"control_comparison{suffix}.png",
    )

    # held-out：只用没参与选层的 5 个 prompt 做最终评估。
    heldout = {
        f"{alpha:+.0f}": evaluate_prompts(
            model, tokenizer, HELDOUT_PROMPTS, best_layer, alpha, best_vector, device
        )
        for alpha in (-8.0, 0.0, 8.0)
    }
    heldout_random = {
        f"{alpha:+.0f}": evaluate_prompts(
            model, tokenizer, HELDOUT_PROMPTS, best_layer, alpha, random_vector, device
        )
        for alpha in (-8.0, 8.0)
    }
    print(f"\nheld-out at layer {best_layer}: {heldout}")
    print(f"held-out random at layer {best_layer}: {heldout_random}")

    save_json(
        {
            "heldout_prompts": HELDOUT_PROMPTS,
            "train_prompts": TRAIN_PROMPTS,
            "full_best": full_best,
            "train_best": train_best,
            "heldout": heldout,
            "heldout_random": heldout_random,
        },
        RESULTS / f"heldout_results{suffix}.json",
    )

    # 三个 prompt 的 top-k 示例。
    examples = []
    for prompt in PROMPTS[:3]:
        for alpha in (-8.0, 0.0, 8.0):
            top = top_next_tokens(
                model, tokenizer, prompt, best_layer, alpha, best_vector, device
            )
            examples.append(
                {
                    "prompt": prompt,
                    "layer": best_layer,
                    "alpha": alpha,
                    "top_next_tokens": [
                        {"token": token, "prob": prob} for token, prob in top
                    ],
                }
            )
            print(f"\n[prompt] {prompt} + ' really' (layer {best_layer}, alpha {alpha:+.1f})")
            print("  " + " | ".join(f"{token!r} {prob:.3f}" for token, prob in top))

    save_json(examples, RESULTS / f"next_token_examples{suffix}.json")

    save_json(
        {
            "device": device,
            "model": "gpt2",
            "pairs": len(PAIRS),
            "prompts": PROMPTS,
            "train_prompts": TRAIN_PROMPTS,
            "heldout_prompts": HELDOUT_PROMPTS,
            "layers": layers,
            "tag": tag,
            "alphas": ALPHAS,
            "full_best": full_best,
            "best": train_best,
            "heldout": heldout,
            "heldout_random": heldout_random,
            "real_control": real_control,
            "real_seed_control": real_seed_summary,
            "real_seeds": list(REAL_SEEDS),
            "random_control": random_control,
            "shuffle_control": shuffle_summary,
            "shuffle_seeds": list(SHUFFLE_SEEDS),
        },
        RESULTS / f"summary{suffix}.json",
    )

    print()
    for name in (
        f"layer_alpha{suffix}.csv",
        f"layer_alpha{suffix}.png",
        f"layer_alpha_train{suffix}.csv",
        f"layer_alpha_train{suffix}.png",
        f"heldout_results{suffix}.json",
        f"real_seed_control{suffix}.csv",
        f"shuffle_control{suffix}.csv",
        f"control_comparison{suffix}.png",
        f"next_token_examples{suffix}.json",
        f"summary{suffix}.json",
    ):
        print(f"wrote: results/{name}")


if __name__ == "__main__":
    main()
