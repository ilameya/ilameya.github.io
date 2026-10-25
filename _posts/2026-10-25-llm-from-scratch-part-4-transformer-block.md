---
title: "The Transformer Block"
subtitle: "Multi-head attention, an MLP, residual connections and LayerNorm: the four ingredients of a block you can stack a hundred times."
description: "Build the transformer block: multi-head attention, the feed-forward MLP, residual connections and LayerNorm, and see in an experiment why residuals and normalisation matter."
series: llm-from-scratch
series_title: LLMs from Scratch
part: 4
date: 2026-10-25 12:00:00 +0200
permalink: /blog/llm-from-scratch/part-4-transformer-block/
code: https://github.com/ilameya/llm-from-scratch/blob/main/parts/04_transformer_block.py
image: /assets/img/llm-from-scratch/04_transformer_block.png
---

In Part 3 we built one attention head. A real transformer layer, or **block**, adds three more ideas around it:
several heads in parallel, a small neural network applied to every token, and two pieces of plumbing (residual
connections and LayerNorm) that make it possible to stack many blocks without training falling apart.

![A pre-norm transformer block. The residual stream runs straight down on the left. The first branch applies LayerNorm and masked multi-head self-attention and adds the result back to the stream. The second branch applies LayerNorm and an MLP (Linear C to 4C, GELU, Linear 4C to C) and adds the result back.]({{ '/assets/img/llm-from-scratch/04_transformer_block.svg' | relative_url }})
*One transformer block. Input and output have the same shape, so blocks stack like Lego.*

The whole block is five lines:

```python
class Block(nn.Module):
    """One transformer block (pre-norm): communicate (attention), then compute (MLP)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, kv_cache=None, save_attn=False):
        x = x + self.attn(self.ln1(x), kv_cache=kv_cache, save_attn=save_attn)  # residual 1
        x = x + self.mlp(self.ln2(x))                                           # residual 2
        return x
```

(The `kv_cache` and `save_attn` arguments are for generation and plotting; ignore them for now.) Let us go
through the ingredients one by one.

## Ingredient 1: multi-head attention

One head computes one attention pattern. But a token may need several kinds of information at once: the previous
character, the start of the current word, whether we are inside a character's name. So we run **several smaller
heads in parallel** and concatenate their outputs.

![Multi-head attention: the input x of shape T by 128 feeds four heads, each doing attention on 32 dimensions. Their outputs are concatenated back to T by 128 and mixed by a linear projection.]({{ '/assets/img/llm-from-scratch/04_multihead.svg' | relative_url }})
*Four heads of size 32 cost the same as one head of size 128, but can learn four different patterns.*

With `n_embd = 128` and `n_head = 4`, each head works in `head_size = 32` dimensions. Instead of creating four
separate modules we compute queries, keys and values for all heads with **one** linear layer and then reshape:

```python
q, k, v = self.qkv(x).split(C, dim=2)                          # each (B, T, C)
# Split the channels into H heads of size D and move heads next to batch
q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, D)
k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, D)
v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, D)
```

After the `transpose`, the head dimension `H` sits next to the batch dimension, so PyTorch treats every head of
every sequence as an independent attention problem: the same code as Part 3, just with an extra dimension.

```python
att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)    # (B, H, T, S)
att = att.masked_fill(~self.mask[:, :, start:start + T, :S], float("-inf"))
att = F.softmax(att, dim=-1)                                   # rows sum to 1
att = self.attn_drop(att)
y = att @ v                                                    # (B, H, T, D)
y = y.transpose(1, 2).contiguous().view(B, T, C)               # concat heads
return self.resid_drop(self.proj(y))
```

The last line reshapes the heads back side by side (that *is* the concatenation) and applies an output projection
`proj`, which lets the heads' results mix.

> **Note.** PyTorch has a fused function, `F.scaled_dot_product_attention(q, k, v, is_causal=True)`, that computes
> exactly these lines, much faster and with less memory (it uses FlashAttention when available). We write it out
> by hand so we can inspect the weights; in your own projects, use the fused version.

## Ingredient 2: the MLP ("feed-forward network")

Attention lets tokens **communicate**. After gathering information, each token needs to **compute** something
with it. That is the job of a small two-layer neural network, applied to every position separately (with the same
weights):

```python
class MLP(nn.Module):
    """Position-wise feed-forward network: each token is processed independently."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)   # expand
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)  # contract
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))
```

It expands each vector to 4 times the width, applies a non-linearity, and projects back. Without the non-linearity
(GELU, a smooth cousin of ReLU) the two layers would collapse into a single matrix, and stacking blocks would add
no expressive power.

A useful mental model: **attention moves information between positions; the MLP transforms information within a
position.** Many researchers think of the MLPs as the place where much of a model's "knowledge" is stored. They
also hold most of the parameters:

```text
Parameters in one block with n_embd=128:
  attention (qkv + proj):  66,048   (~4*C^2 = 65,536)
  MLP (expand + contract): 131,712   (~8*C^2 = 131,072)
  2 x LayerNorm          :     512
  total                  : 198,272   (~12*C^2 = 196,608)
```

## Ingredient 3: residual connections

Look again at the forward pass: `x = x + self.attn(...)`, not `x = self.attn(...)`. Each sub-layer computes a
*change* that is **added** to its input. The vector flowing down the left side of the diagram, which every block
reads from and writes to, is called the **residual stream**.

Why does this help? Two reasons:

- **Gradients have a highway.** During backpropagation the gradient of `x + f(x)` with respect to `x` always
  contains the identity, so the training signal reaches the first layers undiminished, however deep the network.
- **Blocks start out harmless.** At initialisation `f(x)` is small, so each block is close to the identity, and the
  deep network behaves like a shallow one. Training then gradually adds useful changes.

Residual connections come from ResNets in computer vision (He et al., 2015), where they made networks with more
than 100 layers trainable for the first time.

## Ingredient 4: LayerNorm

LayerNorm normalises each token's vector to mean 0 and standard deviation 1, and then applies a learned scale and
shift:

$$ \text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta $$

where $$\mu$$ and $$\sigma$$ are computed over the $$C$$ channels of that one token.

```text
before LayerNorm: mean 2.20, std 4.31
after  LayerNorm: mean 0.00, std 1.00
```

This keeps the numbers going into attention and the MLP in a predictable range, whatever happened in earlier
layers, which makes training faster and more stable.

**Pre-norm vs. post-norm.** The original 2017 Transformer normalised *after* adding the residual:
`x = LayerNorm(x + f(x))`. GPT-2 and nearly every modern LLM normalise the input of each sub-layer instead:
`x = x + f(LayerNorm(x))`. Pre-norm keeps the residual stream itself untouched, which trains more reliably in
deep models. That is why we also need one final LayerNorm at the end of the model (Part 5).

## An experiment: what if we remove them?

Are residuals and LayerNorm really necessary, or just tradition? `parts/04_transformer_block.py` trains the same
8-layer model three times for 800 steps: the full block, a block without LayerNorm, and a block without
residual connections.

![Validation loss curves for three 8-layer models.]({{ '/assets/img/llm-from-scratch/04_block_ablation.png' | relative_url }})
*Same data, same number of parameters, same optimiser. Only the wiring differs.*

```text
--- training 8-layer model: full block
    final val loss 2.175
--- training 8-layer model: no LayerNorm
    final val loss 2.382
--- training 8-layer model: no residual connections
    final val loss 3.352
character-frequency baseline (no context at all): 3.347
```

- **Without LayerNorm** the model still trains, but more slowly: after 800 steps it is clearly behind (2.38 vs.
  2.18). In deeper or longer runs, unnormalised activations also make training much more fragile.
- **Without residual connections** training stalls almost immediately at a loss of 3.35. That number is not
  random. A "model" that ignores the context completely and only knows how often each character appears (`e` and
  space are common, `z` is rare) scores 3.347 on our validation text (the dotted line). So the 8-layer network without residuals
  learned character frequencies and nothing more: the gradient signal could not get through eight stacked
  sub-layers to teach it anything about context.

The residual stream is what makes depth usable. Every modern LLM, with dozens or over a hundred layers, relies on
it.

## Try it yourself

> **Try it:** Change `n_head` from 4 to 1 and to 16 (keeping `n_embd = 128`) and train for 1,000 steps. Does the
> number of heads matter at this scale?

> **Exercise:** Implement a *post-norm* block (`x = ln(x + attn(x))`) and add it to the ablation. How does it
> compare to pre-norm with 8 layers?

> **Exercise:** Replace `F.gelu` with `F.relu`, and then with no activation at all. Explain the result of the
> last version.

## Key takeaways

- A transformer block = **attention** (tokens communicate) + **MLP** (each token computes), each wrapped in
  a **residual connection** with a **LayerNorm** in front.
- Multi-head attention splits the channels into several small heads that learn different patterns.
- Residual connections give gradients a direct path and let deep stacks start out close to the identity.
- Input and output of a block have the same shape `(B, T, C)`, so we can stack as many as we like.

**Next:** [Part 5: Assembling GPT]({{ '/blog/llm-from-scratch/part-5-gpt-model/' | relative_url }}), where we
stack these blocks into the complete model.

### Further reading

- He et al. (2015), [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385).
- Ba, Kiros and Hinton (2016), [*Layer Normalization*](https://arxiv.org/abs/1607.06450).
- Xiong et al. (2020), [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745): why pre-norm trains better.
- Geva et al. (2021), [*Transformer Feed-Forward Layers Are Key-Value Memories*](https://arxiv.org/abs/2012.14913).
