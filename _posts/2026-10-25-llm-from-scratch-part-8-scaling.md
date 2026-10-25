---
title: "Scaling Up, and Where to Go Next"
subtitle: "Our model and ChatGPT share an architecture. What separates them is scale, a few architectural upgrades, and post-training."
description: "A mini scaling experiment on a laptop, scaling laws and compute, what modern LLMs (Llama, Mistral, Qwen) change in the architecture, how a base model becomes an assistant, and project ideas."
series: llm-from-scratch
series_title: LLMs from Scratch
part: 8
date: 2026-10-25 16:00:00 +0200
permalink: /blog/llm-from-scratch/part-8-scaling/
code: https://github.com/ilameya/llm-from-scratch/blob/main/parts/08_scaling.py
image: /assets/img/llm-from-scratch/08_roadmap.png
---

Over seven parts we built a complete GPT: a tokenizer, embeddings, attention, transformer blocks, a training loop
and a sampler, under 700 lines of commented Python. Our model has 0.8 million parameters and writes something
that looks like Shakespeare from a distance.

GPT-3 has 175 *billion* parameters. Modern assistants are larger still, and they answer questions instead of
writing pseudo-Shakespeare. This last part is about the distance between the two, and how much of it you can
already explain.

![Roadmap from our tiny GPT through pretraining at scale, instruction tuning and preference tuning to an assistant, with a panel of architecture upgrades used by modern LLMs: RoPE, RMSNorm, SwiGLU, grouped-query attention, FlashAttention and mixture of experts.]({{ '/assets/img/llm-from-scratch/08_roadmap.svg' | relative_url }})
*Same architecture, same next-token loss, very different scale.*

## A scaling experiment on a laptop

The most important empirical fact about language models fits in one sentence: **bigger models trained on more
data reach lower loss, predictably.** We can see the beginning of this on a laptop.

`parts/08_scaling.py` trains four models of increasing size with exactly the same data, batch size and number of
steps (1,500 steps, dropout off):

```python
sizes = [(2, 32), (3, 64), (4, 128), (6, 192)]  # (n_layer, n_embd)
```

| Layers × width | Parameters | Train loss | Validation loss | Time (2-core CPU) |
|---|---|---|---|---|
| 2 × 32 | 27,552 | 2.35 | 2.37 | 1.9 min |
| 3 × 64 | 154,240 | 2.01 | 2.08 | 1.6 min |
| 4 × 128 | 801,664 | 1.59 | 1.77 | 4.2 min |
| 6 × 192 | 2,682,048 | 1.42 | 1.62 | 11.0 min |

*Parameters excluding the position embedding. Each run sees the same 6.1 million training tokens (1,500 × 32 × 128).
Times are rough: other jobs shared the CPU during these runs.*

![Validation loss against number of parameters on a log axis for four models.]({{ '/assets/img/llm-from-scratch/08_scaling.png' | relative_url }})
*Each model trained for the same number of steps on the same data.*

Three things to notice:

1. **On a log scale, the first three points lie almost on a straight line.** Each roughly 5× increase in
   parameters buys a similar drop in loss, about 0.3. That regularity, not any single number, is what scaling
   laws are about.
2. **The returns shrink at the top.** The last step (3.3× more parameters) gains only 0.14, less than the earlier
   steps would predict, and it costs the most time.
3. **The gap between train and validation loss grows** from 0.02 for the smallest model to 0.21 for the largest.
   The bigger models are starting to memorise Tiny Shakespeare instead of learning general patterns. With only
   about a million characters of text, data, not model size, is becoming our bottleneck.

## Scaling laws

In 2020, researchers at OpenAI trained hundreds of models and found that the loss follows smooth **power laws** in
the number of parameters $$N$$, the amount of training data $$D$$ and the compute $$C$$ (Kaplan et al., 2020).
Plot loss against any of them on log-log axes and you get nearly straight lines over many orders of magnitude.
This is why labs could predict how good a model would be *before* training it, and why they kept building bigger
ones.

A useful approximation: training costs about

$$ C \approx 6 \, N \, D \quad \text{floating-point operations,} $$

two for each parameter and token in the forward pass and four in the backward pass. Let us compare:

| | Parameters $$N$$ | Training tokens $$D$$ | Compute $$6ND$$ |
|---|---|---|---|
| Our model (Part 6) | 0.82M | 12.3M (3,000 steps × 32 × 128) | $$\approx 6 \times 10^{13}$$ |
| GPT-3 (2020) | 175B | 300B | $$\approx 3 \times 10^{23}$$ |

GPT-3 used about **five billion times** more compute than our laptop run. Your laptop does roughly
$$10^{11}$$ to $$10^{12}$$ useful operations per second; GPT-3's training would take it many thousands of
years.

In 2022, DeepMind's *Chinchilla* paper (Hoffmann et al.) refined the recipe: for a fixed compute budget, the best
results come from scaling parameters and data **together**, with roughly **20 training tokens per parameter**.
Many earlier models, GPT-3 included, were too big for the amount of data they saw. (By that rule our 2.7M-parameter
run was badly undertrained: it saw only about 2 tokens per parameter.) Today's models are often
trained far beyond 20 tokens per parameter, because a smaller model that has seen more data is cheaper to *use*.

Our own experiment also shows the data side of the story: Tiny Shakespeare is only about one million characters.
A model that is large enough will start to memorise it, and then the validation loss stops improving even while
the training loss keeps falling. More parameters only help if there is enough data to fill them.

## What modern LLMs change in the architecture

If you read the code of Llama, Mistral or Qwen, you will recognise almost everything. These are the main
differences from our GPT-2-style model:

**RoPE (rotary position embeddings).** Instead of adding a learned position vector to the input, RoPE *rotates*
the query and key vectors by an angle that depends on their position. The dot product $$q_i \cdot k_j$$ then
depends only on the distance $$i - j$$, which generalises better to long contexts. (Su et al., 2021)

**RMSNorm.** A simpler LayerNorm that only rescales and does not subtract the mean. It is cheaper and works just as
well:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

**SwiGLU MLP.** A *gated* MLP: one projection decides how much of another projection to let through.
(Shazeer, 2020) Both `RMSNorm` and this MLP are in `tinyllm/modern.py`, ready for you to swap in.

```python
def forward(self, x):                       # three Linear layers instead of two
    return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))
```

**Grouped-query attention (GQA).** Several query heads share one key/value head. Quality stays almost the same,
but the KV cache from Part 7 gets several times smaller, which matters a lot when serving long conversations.
(Ainslie et al., 2023)

**FlashAttention.** Not a new model, but a smarter way to compute exactly the same attention on a GPU, in tiles
that fit in fast on-chip memory, without ever storing the full $$T \times T$$ matrix. (Dao et al., 2022)

**Mixture of Experts (MoE).** Replace the single MLP in each block with many "expert" MLPs and a small *router*
that sends each token to only one or two of them. The model can have far more parameters while each token still
uses only a fraction of them. Mixtral, DeepSeek-V3 and many recent models work this way. (Shazeer et al., 2017;
Jiang et al., 2024)

**Bigger tokenizers and longer context.** Vocabularies of 32k to 200k BPE tokens instead of our 65 characters,
and context windows of hundreds of thousands of tokens instead of 128.

None of these change the big picture you already know: embed tokens, run a stack of attention + MLP blocks with
residual connections, predict the next token.

## From a base model to an assistant

A model trained only to predict the next token on internet text, a **base model**, is a document completer. Ask
it a question and it might continue with three more questions, because that is what follows a question in many
documents. Assistants get two more training stages:

1. **Supervised fine-tuning (SFT), or instruction tuning.** Continue training on example conversations: a prompt
   and a good answer, written or curated by people. Same loss as ours, different data. The model learns the
   *format* of being helpful.
2. **Preference tuning.** Show the model pairs of answers where people preferred one over the other, and push the
   model towards the preferred ones. The classic method is RLHF (reinforcement learning from human feedback,
   Ouyang et al., 2022); a simpler and now very popular alternative is DPO (direct preference optimisation,
   Rafailov et al., 2023).

The knowledge mostly comes from pretraining; these stages shape behaviour. Both are well within reach of a student
project on a small model.

## Where to go next: project ideas

Pick one and you will learn more than from reading ten more blog posts.

1. **Train on your own data.** Python code, recipes, a novel in the public domain, or text in your own language.
   For non-English text switch to the BPE tokenizer. What changes in the samples and in the loss?
2. **Implement RoPE** in `CausalSelfAttention` and remove `pos_emb`. Compare validation loss with the same
   number of steps.
3. **Swap in RMSNorm and SwiGLU** and turn our GPT-2 into a mini Llama. Keep the parameter count equal for a fair
   comparison.
4. **Load real GPT-2 weights.** Our module structure maps almost one-to-one onto OpenAI's released checkpoint
   (watch out: their linear layers store weights transposed). Then generate with our own `generate()`.
5. **Fine-tune for a task.** Take your trained Shakespeare model and fine-tune it on a few hundred lines of one
   character. How few steps does it take before it only speaks like Juliet?
6. **Measure energy.** Time and measure the power of your training runs at different sizes. How does energy per
   token scale with parameters?

## The series in one paragraph

Text is split into tokens and mapped to ids (Part 1). Ids select learned vectors, and positions add their own
(Part 2). Attention lets each token gather information from earlier tokens with learned queries, keys and values
(Part 3). Blocks combine attention and an MLP with residual connections and LayerNorm (Part 4). A stack of blocks
plus an output layer is GPT (Part 5). Training nudges every weight to make the true next token more likely
(Part 6). Sampling turns the predicted distributions into text, one token at a time (Part 7). Scale the whole thing
up by many orders of magnitude, add post-training, and you have a modern LLM (Part 8).

Thank you for following along. If this series helped you, star the
[repository](https://github.com/ilameya/llm-from-scratch), share it with your study group, and open an issue
with questions or with what you built.

## Key takeaways

- Loss improves predictably with parameters, data and compute (**scaling laws**); compute ≈ $$6ND$$.
- Parameters and data must grow together (**Chinchilla**: roughly 20 tokens per parameter for compute-optimal
  training).
- Modern LLMs keep the GPT recipe and upgrade parts: RoPE, RMSNorm, SwiGLU, GQA, FlashAttention, MoE.
- Assistants are base models plus **instruction tuning** and **preference tuning** (RLHF or DPO).

### Further reading

- Kaplan et al. (2020), [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361).
- Hoffmann et al. (2022), [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556) (Chinchilla).
- Touvron et al. (2023), [*LLaMA: Open and Efficient Foundation Language Models*](https://arxiv.org/abs/2302.13971).
- Su et al. (2021), [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864).
- Ouyang et al. (2022), [*Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155) (InstructGPT).
- Rafailov et al. (2023), [*Direct Preference Optimization*](https://arxiv.org/abs/2305.18290).
- Sebastian Raschka, [*Build a Large Language Model (From Scratch)*](https://github.com/rasbt/LLMs-from-scratch): a book-length version of this journey, including fine-tuning.
