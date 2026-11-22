"""CAA 的核心操作：数据、方向、hook、单点打分。"""

from __future__ import annotations

import os
import random
import sys

# 默认离线，避免每次加载模型都去连 Hugging Face。
if "--online" not in sys.argv:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

PAIRS = [
    ("I thought the movie was wonderful.", "I thought the movie was terrible."),
    ("The film felt great.", "The film felt awful."),
    ("This story was amazing.", "This story was boring."),
    ("Overall, the book was fantastic.", "Overall, the book was dull."),
    ("The experience was enjoyable.", "The experience was horrible."),
    ("The acting was brilliant.", "The acting was bad."),
    ("The plot was delightful.", "The plot was dreadful."),
    ("The direction was excellent.", "The direction was poor."),
    ("The show was fun.", "The show was tedious."),
    ("The ending was satisfying.", "The ending was frustrating."),
    ("I found the characters charming.", "I found the characters annoying."),
    ("The dialogue was witty.", "The dialogue was stupid."),
    ("The visuals were stunning.", "The visuals were ugly."),
    ("The pacing was perfect.", "The pacing was slow."),
    ("The soundtrack was beautiful.", "The soundtrack was noisy."),
    ("The humor was clever.", "The humor was silly."),
    ("The message was moving.", "The message was empty."),
    ("The premise was promising.", "The premise was ridiculous."),
    ("The cast was talented.", "The cast was wooden."),
    ("I would recommend it.", "I would avoid it."),
]

PROMPTS = [
    "The movie was",
    "I thought the film was",
    "This story felt",
    "Overall, the book was",
    "The show was",
    "The acting in it was",
    "The plot seemed",
    "The ending felt",
    "The direction was",
    "The dialogue seemed",
    "The characters were",
    "The pacing felt",
    "The visuals were",
    "The soundtrack was",
    "The experience was",
    "The whole thing was",
    "I found the film",
    "The first half was",
    "The second half was",
    "Looking back, the movie was",
]

HELDOUT_PROMPTS = [
    "The show was",
    "The plot seemed",
    "The characters were",
    "The experience was",
    "Looking back, the movie was",
]
TRAIN_PROMPTS = [p for p in PROMPTS if p not in HELDOUT_PROMPTS]

PREFIX = " really"
POS_WORD = " great"
NEG_WORD = " awful"


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(device: str):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    return model, tokenizer


def last_token_hidden(model, tokenizer, text: str, layer: int, device: str) -> torch.Tensor:
    """第 layer 个 Transformer block 之后、最后一个 token 的 hidden state。"""
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return out.hidden_states[layer + 1][0, -1, :].detach()


def build_direction(
    model,
    tokenizer,
    layer: int,
    device: str,
    shuffle_seed: int | None = None,
    bootstrap_seed: int | None = None,
) -> torch.Tensor:
    """正负句子最后一个 token 的 difference-in-means，归一化。"""
    pairs = PAIRS
    if bootstrap_seed is not None:
        rng = random.Random(bootstrap_seed)
        pairs = [PAIRS[rng.randrange(len(PAIRS))] for _ in range(len(PAIRS))]
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        pairs = [(neg, pos) if rng.random() < 0.5 else (pos, neg) for pos, neg in pairs]

    diffs = [
        last_token_hidden(model, tokenizer, pos, layer, device)
        - last_token_hidden(model, tokenizer, neg, layer, device)
        for pos, neg in pairs
    ]
    vector = torch.stack(diffs).mean(dim=0)
    return vector / vector.norm().clamp_min(1e-8)


def add_vector_hook(vector: torch.Tensor, alpha: float, start: int):
    """给位置 >= start 的 token 加 alpha * vector。"""

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            if alpha != 0.0 and hidden.shape[1] > start:
                hidden[:, start:, :] += alpha * vector
            return (hidden,) + output[1:]

        hidden = output.clone()
        if alpha != 0.0 and hidden.shape[1] > start:
            hidden[:, start:, :] += alpha * vector
        return hidden

    return hook


def forward_with_hook(model, input_ids, layer: int, alpha: float, vector, start: int):
    """临时在 block layer 上挂 hook，跑一次 forward。"""
    handle = model.transformer.h[layer].register_forward_hook(
        add_vector_hook(vector, alpha, start)
    )
    try:
        with torch.no_grad():
            return model(input_ids=input_ids)
    finally:
        handle.remove()


def sentiment_logp_diff(model, tokenizer, prompt: str, layer: int, alpha: float, vector, device: str) -> float:
    """log P(' great' | prompt + ' really') - log P(' awful' | ...)。"""
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = tokenizer(prompt + PREFIX, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    start = prompt_ids.shape[1]

    if full_ids.shape[1] <= start:
        raise ValueError(f"Prefix tokenized to zero tokens: {PREFIX!r}")
    if not torch.equal(full_ids[:, :start], prompt_ids):
        raise ValueError(f"Prompt/prefix tokenization mismatch: {prompt!r}")

    pos_ids = tokenizer.encode(POS_WORD, add_special_tokens=False)
    neg_ids = tokenizer.encode(NEG_WORD, add_special_tokens=False)
    if len(pos_ids) != 1 or len(neg_ids) != 1:
        raise ValueError("Sentiment words must each tokenize to one token")

    logits = forward_with_hook(model, full_ids, layer, alpha, vector, start).logits[0, -1]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return float((log_probs[pos_ids[0]] - log_probs[neg_ids[0]]).item())


def top_next_tokens(model, tokenizer, prompt: str, layer: int, alpha: float, vector, device: str, k: int = 8):
    """prompt + ' really' 之后概率最高的 k 个 token。"""
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = tokenizer(prompt + PREFIX, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    start = prompt_ids.shape[1]

    if not torch.equal(full_ids[:, :start], prompt_ids):
        raise ValueError(f"Prompt/prefix tokenization mismatch: {prompt!r}")

    logits = forward_with_hook(model, full_ids, layer, alpha, vector, start).logits[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, k=k)
    return [
        (tokenizer.decode([int(token_id)]), float(prob))
        for token_id, prob in zip(top.indices, top.values)
    ]
