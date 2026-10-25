---
title: "Generating Text: Sampling and the KV Cache"
subtitle: "The model only gives us probabilities. How we pick from them decides whether the output is boring, creative or gibberish."
description: "Autoregressive generation, greedy decoding, temperature, top-k and top-p (nucleus) sampling, with real outputs from our trained model, plus the KV cache that makes generation fast."
series: llm-from-scratch
series_title: LLMs from Scratch
part: 7
date: 2026-10-25 15:00:00 +0200
permalink: /blog/llm-from-scratch/part-7-sampling/
code: https://github.com/ilameya/llm-from-scratch/blob/main/tinyllm/sample.py
image: /assets/img/llm-from-scratch/07_temperature.png
---

We have a trained model. Given some text, it returns a probability for every possible next character. To write a
whole passage we use the loop from Part 1: predict, pick a character, append it, predict again. This is called
**autoregressive generation**, and the interesting part is the word *pick*.

```bash
python parts/07_sampling.py
python -m tinyllm.sample --prompt "JULIET:" --temperature 0.8 --top_p 0.9
```

## The generation loop

```python
for _ in range(max_new_tokens):
    context = idx[:, -block_size:]                       # crop to the context window
    logits, _ = model(context)                           # (B, T, V)
    nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p, greedy, generator)
    idx = torch.cat([idx, nxt], dim=1)
```

Three details:

- We only use the logits of the **last position**, `logits[:, -1, :]`. The predictions for earlier positions were
  needed during training, not now.
- The model has no row in its position table beyond `block_size`, so we **crop** the context to the last 128
  characters. The model simply forgets anything older.
- `sample_next` decides which token to take. Everything else in this post is about that one function.

Let us look at what the model actually predicts for the prompt `"ROMEO:\nI will "`:

```text
after 'ROMEO:\nI will ' the model predicts:
  't'    0.150
  'n'    0.091
  's'    0.077
  'h'    0.071
  'b'    0.068
  'm'    0.063
  'a'    0.059
  'w'    0.056
```

The model is far from certain: its favourite, `t`, gets only 15%. Plenty of letters could start the next
word (`the`, `not`, `see`, `have`, `be`, `make`...), and the model spreads its bets accordingly.

## Strategy 1: greedy decoding

The obvious choice: always take the most likely token.

```python
if greedy:
    return logits.argmax(dim=-1, keepdim=True)
```

```text
ROMEO:
I will the shall the see the see the see the see
That the shall the see the see the see the see
To see the see the see the see the see the see
To see the see
```

Greedy decoding is deterministic, and it tends to fall into **loops**. Once a phrase is likely, repeating it makes
it even more likely. Always choosing the single best next step does not produce the best text overall; it
produces the blandest.

## Strategy 2: sampling, with a temperature

Instead, we can **sample**: draw the next token at random, *according to* the model's probabilities. A token with
probability 0.3 is chosen 30% of the time. To control how adventurous this is, we divide the logits by a
**temperature** $$\tau$$ before the softmax:

$$ p_i = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}} $$

- $$\tau < 1$$ sharpens the distribution: likely tokens become even more likely (safer, more repetitive).
- $$\tau = 1$$ uses the model's probabilities as they are.
- $$\tau > 1$$ flattens it: unlikely tokens get a real chance (more surprising, more mistakes).
- $$\tau \to 0$$ becomes greedy decoding.

![Three bar charts of the ten most likely next characters after "I will " at temperatures 0.5, 1.0 and 1.5.]({{ '/assets/img/llm-from-scratch/07_temperature.png' | relative_url }})
*The same prediction at three temperatures. Only the shape changes, never the order.*

Here is the same prompt, sampled with the same random seed at three temperatures:

**Temperature 0.5**

```text
ROMEO:
I will not man to me: lord, I will you see
The father fair thy for the make him and the shame
What will be war a consul, and the for the to and for
Against a
```

**Temperature 1.0**

```text
ROMEO:
I will leave may commolor,
If fase your eath mour
Reable very my rature born to weep, and is a charping?

ESABELUS:
Dowly, their old may the seck, we warrnes
```

**Temperature 1.5**

```text
ROMEO:
I will luk's not
cos: lord, lett'er wissester toG
RUT:
Fore thmer stout roadh whate's festing och'd.

RICHARARD:
Reemnland:
God are king; to 'tis, year rage
```

At 0.5 the text is cautious: real words, many repeats of safe choices like `the` and `and`. At 1.0 we see the
model's true distribution: livelier, with invented words (`commolor`, `rature`) and a new speaker. At 1.5 spelling
falls apart (`wissester toG`, `RICHARARD`), because rare, wrong characters now get picked regularly.

## Strategy 3: top-k and top-p

Pure sampling has a weakness: the model gives a tiny probability to each of dozens of bad tokens, and together
those tiny probabilities add up. Every so often one gets picked and the text derails. Two simple filters cut off
this long tail before sampling.

**Top-k** keeps only the `k` most likely tokens. **Top-p**, also called *nucleus sampling*, keeps the smallest set
of tokens whose probabilities add up to at least `p`. Top-p adapts to the situation: when the model is confident,
the nucleus might be one or two tokens; when many continuations are plausible, it grows.

```python
def filter_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                  top_p: float | None = None) -> torch.Tensor:
    """Apply temperature, top-k and top-p (nucleus) filtering to (B, V) logits."""
    logits = logits / max(temperature, 1e-8)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth_best = torch.topk(logits, k).values[:, [-1]]                 # (B, 1)
        logits = logits.masked_fill(logits < kth_best, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Drop tokens once the probability mass BEFORE them already exceeds top_p
        drop = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits
```

Filtered tokens get a logit of $$-\infty$$, so after the softmax their probability is exactly 0, the same trick as
the causal mask in Part 3.

**Top-k = 10** (temperature 1.0)

```text
ROMEO:
I will not man's so wolds,
I have so was any more, table we dam, who must a have
temper ships of the worlden that we made at the flight,
What's for have age
```

**Top-p = 0.9** (temperature 1.0)

```text
ROMEO:
I will not man's common,
So lifes all sue them father fanter your storm boody
what have shame of thy wish to that were here beggarl they to
Was I need to go
```

Both keep the energy of temperature 1.0 while avoiding the worst derailments.

**In practice**, a temperature around 0.7 to 1.0 combined with top-p around 0.9 is a common default for chat
models; lower temperatures are typical for code or factual answers, where creativity is a bug. Fixing the random
seed (`--seed`) makes any of these strategies reproducible.

## Making generation fast: the KV cache

Look at the generation loop again. To produce token number 100, we run the model over all 100 tokens, then throw
away everything except the last position's output. To produce token 101, we run it over 101 tokens, again. Most of
that work is repeated.

The repeated work is in attention. When a new token arrives, the **keys and values of all earlier tokens are
exactly the same as last step** (thanks to the causal mask, earlier positions never depend on later ones). So we
can store them and only compute the query, key and value of the newest token.

![Without a KV cache, every generation step recomputes all tokens so far: 1 + 2 + 3 + 4 + 5 = 15 token passes. With a KV cache, earlier keys and values are read from the cache and only the newest token is computed: 5 token passes.]({{ '/assets/img/llm-from-scratch/07_kv_cache.svg' | relative_url }})
*Generating "ROMEO" one character at a time, without and with a cache.*

Inside `CausalSelfAttention` this is only a few lines: append the new keys and values to the cache and attend over
the whole thing.

```python
if kv_cache is not None:  # generation: reuse keys/values of earlier tokens
    if "k" in kv_cache:
        start = kv_cache["k"].size(2)
        k = torch.cat([kv_cache["k"], k], dim=2)
        v = torch.cat([kv_cache["v"], v], dim=2)
    kv_cache["k"], kv_cache["v"] = k, v
```

Generation then has two phases: **prefill** (process the whole prompt once and fill the cache) and **decode**
(feed one token at a time):

```python
caches = [dict() for _ in model.blocks]
logits, _ = model(idx, kv_caches=caches)           # "prefill": process the prompt once
for _ in range(max_new_tokens):
    nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p, greedy, generator)
    idx = torch.cat([idx, nxt], dim=1)
    logits, _ = model(nxt, kv_caches=caches)       # "decode": only the new token
```

Does it work? The script generates the same text with and without the cache and times both:

```text
use_cache=False: 114 tokens in 258 ms
use_cache=True: 114 tokens in 102 ms
identical outputs: True | speed-up 2.5x
```

The outputs are identical (our test suite checks this too); only the amount of work changes. The speed-up grows
with the length of the text. For our tiny model and short context it is modest; for a real model generating
thousands of tokens it is the difference between usable and unusable.

The price is **memory**. The cache stores a key and a value vector for every layer and every token:
$$2 \times n_\text{layer} \times T \times C$$ numbers per sequence. For GPT-2 small with a full 1,024-token context
that is $$2 \times 12 \times 1024 \times 768 \approx 19$$ million numbers, about 75 MB in 32-bit floats, *per
conversation*. This is why serving LLMs is so memory-hungry, and why tricks like grouped-query attention (Part 8)
exist to shrink the cache.

## Try it yourself

> **Try it:** Generate 500 characters at temperature 0.2, 0.8 and 2.0. At what temperature does the text stop
> looking like English? At what temperature does it start repeating itself?

> **Exercise:** Implement **min-p** sampling: keep only tokens whose probability is at least `min_p` times the
> probability of the most likely token. Compare `min_p = 0.1` with `top_p = 0.9`.

> **Exercise:** Our KV cache stops at `block_size`. Extend `generate` so that when the cache is full it keeps
> only the most recent tokens. Why does this not give exactly the same result as cropping the context without a
> cache? (Hint: think about the position embeddings.)

## Key takeaways

- Generation is a loop: run the model, take the **last position's** distribution, pick a token, append.
- **Greedy** decoding is deterministic and repetitive; **sampling** with a **temperature** trades safety for
  variety.
- **Top-k** and **top-p** remove the long tail of unlikely tokens before sampling.
- The **KV cache** stores keys and values of past tokens, so each step only processes the new token: same output,
  much less compute, more memory.

**Next:** [Part 8: Scaling up]({{ '/blog/llm-from-scratch/part-8-scaling/' | relative_url }}), the last part: what
separates our 0.8M-parameter model from the LLMs you use every day.

### Further reading

- Holtzman et al. (2019), [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751): the paper behind top-p sampling and an analysis of why greedy text is bland.
- Pope et al. (2022), [*Efficiently Scaling Transformer Inference*](https://arxiv.org/abs/2211.05102): KV caches and inference at scale.
- Hugging Face, [*How to generate text*](https://huggingface.co/blog/how-to-generate): a visual tour of decoding strategies.
