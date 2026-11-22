# GPT-2 上的 CAA 小实验：情感方向

这是我在 GPT-2 small 上做的一个 CAA 练习。

目标不是复现论文里的 Llama 2 实验，而是亲手验证一件事：从激活里算出来的情感方向，加回去能不能改变模型输出。

代码、结果和运行方式都在这个仓库里。

## 我做了什么

1. 构造 20 对正负情感句子，每对只差一个形容词，比如 `wonderful` 和 `terrible`。
2. 取每句最后一个 token 在第 L 层的 hidden state，算正负样本的平均差，得到方向 `v`，再归一化。
3. 生成时在 prompt 之后的 token 位置加 `αv`，prompt 本身不加。
4. 用 `" really"` 当第一个生成 token，比较下一个 token 的概率：

```text
ΔlogP = logP(" great" | prompt + " really")
       - logP(" awful" | prompt + " really")
```

5. 扫了 L7、L8、L9 三层，α 取 -8、-4、0、+4、+8。
6. 用 20 个中性开头做评估，其中 5 个留作 held-out，不参与选层和选 α。
7. 做了四组对照：真实方向、bootstrap 重采样、shuffle-label、随机方向。
8. 在 WikiText-2 的 8 个 512-token 文本块上测了语言建模 loss，看 steering 会不会损伤一般能力。

## 结果

### 主结果

表中是 20 个 prompt 的平均 ΔlogP。

| 层 | α=-8 | α=0 | α=+8 |
|---|---:|---:|---:|
| L7 | +2.17 | +3.38 | +4.48 |
| L8 | +2.31 | +3.38 | +4.48 |
| L9 | +2.47 | +3.38 | +4.31 |

20 个 prompt 全部同向。α 从 -8 到 +8，ΔlogP 单调上升。

L7 和 L8 基本一样，L9 开始下降。粗扫里的 L4、L6、L10 都比 L7/L8 弱。

L8 在全量 20 个 prompt 上最好，L7 在 15 个 train prompt 上最好，两者差距很小。

### 对照

在 L7 上比较：

| 方向 | α=-8 | α=+8 | 摆动 |
|---|---:|---:|---:|
| 真实方向 | +2.17 | +4.48 | 2.31 |
| 真实方向 bootstrap | +2.21 | +4.45 | 2.23 |
| shuffle-label | +3.29 | +3.42 | 0.13 |
| 随机方向 | +3.21 | +3.54 | 0.33 |

真实方向的摆动明显大于 shuffle-label 和随机方向。

shuffle-label 单个 seed 会波动，5 个 seed 平均后基本回到基线，所以这个对照要看平均值。

### held-out

用 15 个 train prompt 选层和 α，再在 5 个没参与选择的 prompt 上评估。

| 评估集 | α=-8 | α=0 | α=+8 | +8 相对 0 |
|---|---:|---:|---:|---:|
| train 15 个 | +2.02 | +3.24 | +4.35 | +1.11 |
| held-out 5 个 | +2.62 | +3.82 | +4.87 | +1.05 |
| held-out 5 个，随机方向 | +3.64 | +3.82 | +3.99 | +0.16 |

held-out 上真实方向仍有 +1.05 的摆动，随机方向只有 +0.16。

held-out 的基线比 train 高，因为这 5 个句式本身更偏正向。这里只看相对 α=0 的摆动，不跨集合比较绝对值。

### 能力损失

baseline 的 loss 是 3.5948，ppl 是 36.41。

| 层 | 条件 | ppl | 相对 baseline |
|---|---|---:|---:|
| L7 | 真实 α=-8 | 37.08 | +1.85% |
| L7 | 真实 α=+8 | 37.46 | +2.88% |
| L7 | 随机 α=+8 | 37.06 | +1.78% |
| L8 | 真实 α=-8 | 36.60 | +0.52% |
| L8 | 真实 α=+8 | 37.11 | +1.92% |
| L8 | 随机 α=+8 | 36.87 | +1.27% |

α=8 时最大的变化是 L7 的 +2.88% ppl，没有出现明显的能力崩塌，但也不是零成本。

shuffle 方向在 L7/L8 的 +8 上分别是 37.26 和 36.83，说明随便加一个方向本身也会带来一点扰动。

## 图怎么看

`results/layer_alpha_fine.png` 是 layer × alpha 热图。颜色是相对 α=0 的变化：蓝色表示往负情感推，红色表示往正情感推。

`results/control_comparison_fine.png` 是真实方向、bootstrap、shuffle-label、随机方向的对照。真实方向明显更高或更低，两个负对照都贴着基线。

`results/capability_loss.png` 是困惑度相对 baseline 的变化。柱子越高，语言建模能力掉得越多。

## 怎么运行

在仓库根目录执行：

```powershell
pip install -r requirements.txt

python caa_toy.py --layers "7,8,9" --tag fine

python caa_toy.py --layers "4,6,8,10" --tag coarse

python capability_eval.py
```

第一次运行需要下载 GPT-2，可以加 `--online`。模型缓存后默认离线运行。

完整跑一遍大约 1 到 2 分钟，取决于 GPU。8 GB 显存够用，CPU 也能跑，只是慢一些。

## 文件

```text
caa_toy.py           入口，命令行和主流程
caa_core.py          数据、模型、方向、hook、打分
caa_experiments.py   扫描、对照、画图
capability_eval.py   能力损失实验
requirements.txt     依赖
data/                固定的 WikiText-2 小文本
results/             三张结果图
```

仓库只提交三张关键图。完整的 CSV、JSON 和日志在本地运行后生成，不会提交。

## 结论和边界

这个实验支持三件事：

- 情感信息在 residual stream 里有一个近似线性的方向。
- 把这个方向加回去，会改变下一个 token 的概率。
- 这个效果不是随便一个向量都能产生的，随机方向和 shuffle-label 都弱很多。

但它不能说明：

- 这不是 CAA 论文复现。模型是 GPT-2 small，不是 Llama 2 7B/13B。
- 只测了下一个形容词 token，不是完整生成质量。
- 20 个 prompt 是手写的，5 个 held-out 也是事后留出的，样本不大。
- 能力损失只测了 WikiText 小样本的语言建模 loss，不是 MMLU 或完整能力评测。
