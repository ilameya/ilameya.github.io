---
title: "Assembling GPT"
subtitle: "We have all the pieces. Now we stack them into a complete model and follow one batch through it, shape by shape."
description: "The full GPT architecture in under 200 lines of PyTorch: embeddings, N transformer blocks, final LayerNorm, the output layer, weight tying, initialisation and parameter counting."
series: llm-from-scratch
series_title: LLMs from Scratch
part: 5
date: 2026-10-25 13:00:00 +0200
permalink: /blog/llm-from-scratch/part-5-gpt-model/
code: https://github.com/ilameya/llm-from-scratch/blob/main/tinyllm/model.py
image: /assets/img/llm-from-scratch/05_gpt_architecture.png
---

In Parts 1 to 4 we built a tokenizer, embeddings, self-attention and the transformer block. Today we snap them
together into the model the series is named after: a **GPT** (Generative Pre-trained Transformer). This is the
same architecture as GPT-2; only the sizes are smaller.

![The full GPT: token ids go into a token embedding and a position embedding, which are added, passed through dropout, N transformer blocks, a final LayerNorm and a linear output layer that produces logits over the vocabulary. The output layer shares its weights with the token embedding.]({{ '/assets/img/llm-from-scratch/05_gpt_architecture.svg' | relative_url }})
*The complete model. Every box is something we have already written.*

## The configuration

All the sizes live in one small dataclass:

```python
@dataclass
class GPTConfig:
    vocab_size: int = 65    # number of distinct tokens
    block_size: int = 128   # maximum context length (tokens the model can "see")
    n_layer: int = 4        # number of transformer blocks
    n_head: int = 4         # attention heads per block
    n_embd: int = 128       # width of the residual stream
    dropout: float = 0.1
    bias: bool = True       # bias terms in Linear / LayerNorm
```

For GPT-2 small the same fields would be `vocab_size=50257, block_size=1024, n_layer=12, n_head=12,
n_embd=768`. We will instantiate exactly that at the end of this post.

## The model

```python
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token id  -> vector
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # position  -> vector
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: the input embedding and the output projection share weights.
        self.lm_head.weight = self.tok_emb.weight
```

And the forward pass, slightly simplified (the full version also supports the KV cache from Part 7):

```python
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)                # (T,)
        x = self.tok_emb(idx) + self.pos_emb(pos)                   # (B, T, C)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                    # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```

That is the whole model. Let us walk through the parts we have not met yet.

### The final LayerNorm

Because our blocks are *pre-norm* (each sub-layer normalises its own input, Part 4), the residual stream itself
is never normalised. One last `LayerNorm` cleans it up before the output layer.

### The output layer (`lm_head`) and logits

`lm_head` is a single linear layer from `C` to `V`: for each position it produces one score per vocabulary
token. These raw scores are called **logits**. A softmax turns them into probabilities:

$$ p(\text{next token} = i) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}} $$

Note that the model makes a prediction **at every position**, not only the last. Position 0 predicts token 1,
position 1 predicts token 2, and so on. During training this gives us `T` training examples per sequence for the
price of one forward pass (Part 6). During generation we only use the last position (Part 7).

### Weight tying

`tok_emb.weight` has shape `(V, C)`: one row per token. `lm_head.weight` also has shape `(V, C)`. The line

```python
self.lm_head.weight = self.tok_emb.weight
```

makes them **the same tensor**. The intuition: if two tokens are similar enough to have similar input
embeddings, they should also get similar output scores. Tying saves `V × C` parameters and usually trains a
little better. GPT-2 does this; many larger models do not, because at their scale the savings are tiny.

### Initialisation

```python
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

Small random weights (standard deviation 0.02) plus one GPT-2 trick: the projections that *write into* the
residual stream (`attn.proj` and `mlp.proj`) are scaled down by $$1/\sqrt{2N}$$, because $$2N$$ of them add
up along the residual stream. Without it, the stream's variance grows with depth.

## Following one batch through the model

`parts/05_gpt_model.py` registers *forward hooks*, small callbacks that print each module's output shape. We feed
it two sequences of 20 characters each:

```text
input ids: (2, 20)   (B=2 sequences, T=20 tokens)
  tok_emb        -> (2, 20, 128)
  pos_emb        -> (20, 128)
  blocks.0.ln1   -> (2, 20, 128)
  blocks.0.attn  -> (2, 20, 128)
  blocks.0.mlp   -> (2, 20, 128)
  blocks.0       -> (2, 20, 128)
  blocks.3       -> (2, 20, 128)
  ln_f           -> (2, 20, 128)
  lm_head        -> (2, 20, 65)
```

Read it top to bottom and you have the whole architecture:

- Ids `(B, T)` become vectors `(B, T, C)`. The position embedding is `(T, C)` and is **broadcast** over the batch:
  every sequence uses the same position vectors.
- Every block keeps the shape `(B, T, C)`. That is why we can stack as many as we like.
- Only the very last layer changes the width, from `C = 128` to `V = 65` scores.

## A sanity check: the loss before training

What loss should an untrained model have? If it knew nothing, it would assign probability $$1/V$$ to every
token, and the cross-entropy loss would be

$$ -\ln\frac{1}{V} = \ln V = \ln 65 \approx 4.17 $$

```text
loss at init: 4.252   ln(V) = ln(65) = 4.174
```

Close. This check is worth doing in every project: if your initial loss is far above $$\ln V$$, something is
wrong with the initialisation (for example, weights too large make the model confidently wrong).

And here is what the untrained model "writes":

```text
untrained sample: 'ROMEO:\nCuw;suqW\n.&TTNUuuOgl,O,cKavjeXLZ&Kf .UK.msowMKPXDcvLf;KvVSKK3 3cOW$Nir.CLIoxZJW'
```

Random characters, as expected. Keep this in mind for Part 6.

## Where do the parameters live?

![Bar chart: MLP 526,848 parameters (64%), attention 264,192 (32%), position embedding 16,384 (2%), token embedding 8,320 (1%), LayerNorms 2,304 (0.3%).]({{ '/assets/img/llm-from-scratch/05_param_breakdown.png' | relative_url }})
*Our default model: 818,048 parameters in total.*

Two lessons:

1. **The MLPs hold two thirds of the parameters.** Each block's MLP has about $$8C^2$$ weights, its attention
   about $$4C^2$$. This is why the rule of thumb for transformer size is

   $$ N_\text{params} \approx 12 \cdot n_\text{layer} \cdot n_\text{embd}^2 $$

   For us: $$12 \times 4 \times 128^2 = 786{,}432$$, within 4% of the true count.
2. **Our embeddings are tiny** because our vocabulary is tiny (65 characters). With GPT-2's 50,257 BPE tokens the
   token embedding alone would be $$50{,}257 \times 768 \approx 38.6$$M parameters, almost a third of the model.

## The same code at GPT-2 scale

```python
gpt2 = GPT(GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768, dropout=0.0))
```

```text
GPT-2 small config -> 124.4M parameters (OpenAI reports 124M)
```

Our 180-line `model.py`, comments included, produces GPT-2 small, exactly. What OpenAI added was not a different architecture but
data (40 GB of web text instead of 1 MB of Shakespeare) and compute.

## Try it yourself

> **Try it:** Remove the weight-tying line, recount the parameters, and train both versions for 1,000 steps
> (`python -m tinyllm.train --steps 1000`). Does tying help at this scale?

> **Exercise:** Without running any code, compute the parameter count for `n_layer=6, n_embd=384` with our
> 65-character vocabulary. Then check your answer with `GPT(GPTConfig(n_layer=6, n_embd=384, n_head=6)).num_params()`.

> **Exercise:** Change the initialisation standard deviation from `0.02` to `1.0`. What is the loss at
> initialisation now, and why is it so much higher than $$\ln V$$?

## Key takeaways

- GPT = token embedding + position embedding → N transformer blocks → final LayerNorm → linear layer to the
  vocabulary.
- The model outputs **logits for every position**; softmax turns them into next-token probabilities.
- An untrained model should start near loss $$\ln V$$. Check this.
- Parameters scale like $$12 L C^2$$, and most of them live in the MLPs.

**Next:** [Part 6: Training]({{ '/blog/llm-from-scratch/part-6-training/' | relative_url }}), where this model
learns to write.

### Further reading

- Radford et al. (2019), [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2).
- Press and Wolf (2017), [*Using the Output Embedding to Improve Language Models*](https://arxiv.org/abs/1608.05859): the weight-tying paper.
- Karpathy's [nanoGPT `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py), which our model closely follows.
