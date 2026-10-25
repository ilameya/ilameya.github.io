---
title: "Self-Attention, Step by Step"
subtitle: "The one idea that makes transformers work: every token decides which earlier tokens to listen to."
description: "Build causal self-attention from a for-loop average to queries, keys, values, the causal mask and the 1/√d scaling, then look at real attention maps from a trained model."
series: llm-from-scratch
series_title: LLMs from Scratch
part: 3
date: 2026-10-25 11:00:00 +0200
permalink: /blog/llm-from-scratch/part-3-attention/
code: https://github.com/ilameya/llm-from-scratch/blob/main/parts/03_attention.py
image: /assets/img/llm-from-scratch/03_attention_flow.png
---

After Part 2, every token is a vector that knows *what* it is and *where* it is. But each vector still lives in
isolation. To predict what comes after `"To be, or not to b"`, the model at the last position needs information
from earlier positions: that we are inside a famous phrase, that the previous word was `to`, that `b` started a
new word.

**Self-attention** is the mechanism that lets tokens exchange information. We will build it in five small steps,
each a few lines of PyTorch. Run along with:

```bash
python parts/03_attention.py
```

Throughout this post `x` has shape `(B, T, C)`: a batch of `B` sequences, `T` tokens each, every token a
`C`-dimensional vector.

## Step 1: the simplest communication, averaging

The simplest way to let token `t` know about the past: replace it with the **average of itself and all tokens
before it**.

```python
avg_loop = torch.zeros_like(x)
for t in range(T):
    avg_loop[:, t] = x[:, : t + 1].mean(dim=1)
```

Note `: t + 1`: token `t` may look at tokens `0 ... t`, **never at the future**. When we train the model to predict
token `t + 1`, letting it peek at token `t + 1` would be cheating, and it would learn nothing useful.

Averaging throws away a lot (order, for one), but it is a start. The problem: the loop is slow.

## Step 2: the same average as one matrix multiplication

Here is the trick that the rest of the post builds on. Make a lower-triangular matrix whose rows sum to 1:

```python
w = torch.tril(torch.ones(T, T))
w = w / w.sum(dim=1, keepdim=True)
avg_matmul = w @ x
```

```text
averaging weights (lower-triangular):
 [[1.   0.   0.   0.  ]
 [0.5  0.5  0.   0.  ]
 [0.33 0.33 0.33 0.  ]
 [0.25 0.25 0.25 0.25]]
loop == matmul? True
```

Row `t` of `w` says *how much each position contributes to output `t`*. Multiplying `w @ x` computes all the
weighted averages at once. Any **weighted** sum of the past can be written this way, if we choose `w` well.

## Step 3: the same thing again, with softmax

One more rewrite. Start from a matrix of scores, set the future to $$-\infty$$, and apply softmax to each row:

```python
scores = torch.zeros(T, T)
scores = scores.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
w_soft = F.softmax(scores, dim=-1)
```

Since $$e^{-\infty} = 0$$, the future gets weight exactly 0, and equal scores give equal weights: we get the same
averaging matrix again (`softmax version == matmul? True`).

Why bother? Because now the weights come from **scores**, and scores can be anything. Right now they are all zero,
so every past token counts equally. If instead the scores depended on the tokens themselves, each token could
choose *which* past tokens matter to it. That is attention.

## Step 4: queries, keys and values

How should a token compute a score for another token? Each token produces three vectors with three learned
linear layers:

- a **query** $$q$$: *what am I looking for?*
- a **key** $$k$$: *what do I contain?*
- a **value** $$v$$: *what do I hand over if someone attends to me?*

The score between token $$i$$ and an earlier token $$j$$ is the dot product $$q_i \cdot k_j$$: large when the
query matches the key. A library analogy: the query is what you type into the search box, the keys are the
titles on the spines, and the values are the contents of the books. You read (a blend of) the books whose titles
match your search.

![Causal self-attention: x is multiplied by three weight matrices to give queries, keys and values. Q times K transposed, scaled by one over root d and masked, gives a T by T score matrix with minus infinity above the diagonal. Softmax per row turns it into attention weights that sum to one, and the weights mix the values into the output.]({{ '/assets/img/llm-from-scratch/03_attention_flow.svg' | relative_url }})
*One attention head. Orange cells are allowed connections; gray cells (the future) are masked.*

In code, one **attention head**:

```python
head_size = 4
query = nn.Linear(C, head_size, bias=False)
key = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
q, k, v = query(x), key(x), value(x)                          # each (B, T, head_size)
scores = q @ k.transpose(-2, -1) / math.sqrt(head_size)       # (B, T, T)
scores = scores.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
weights = F.softmax(scores, dim=-1)
out = weights @ v                                             # (B, T, head_size)
```

With random (untrained) weights, the attention pattern is no longer a plain average:

```text
attention weights (each row sums to 1):
 [[1.   0.   0.   0.  ]
 [0.47 0.53 0.   0.  ]
 [0.51 0.22 0.27 0.  ]
 [0.2  0.21 0.37 0.21]]
```

Training will shape these weights into something useful. The whole step fits in one famous formula from
*Attention Is All You Need*:

$$ \text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right) V $$

where $$M$$ is the causal mask (0 for the past, $$-\infty$$ for the future) and $$d$$ is the head size.

## Step 5: why divide by $$\sqrt{d}$$?

If the entries of $$q$$ and $$k$$ are roughly independent with variance 1, their dot product is a sum of $$d$$
terms, so its variance is $$d$$. We can check this numerically:

```text
d=   4: variance of q.k =     4.2   after / sqrt(d): 1.04
d=  64: variance of q.k =    63.3   after / sqrt(d): 0.99
d= 512: variance of q.k =   512.4   after / sqrt(d): 1.00
```

Large scores are a problem for softmax: it exaggerates differences exponentially, so one score ends up with almost
all the weight. A token then listens to a single other token, and the gradients for everything else become tiny.
Dividing by $$\sqrt{d}$$ keeps the scores at variance 1, no matter how big the head is.

![Two bar charts of attention over five tokens. With scaling the weights are spread: 0.11, 0.15, 0.22, 0.29, 0.24. Without scaling (scores 8 times larger, as for d = 64) the same scores give 0.00, 0.00, 0.07, 0.77, 0.16.]({{ '/assets/img/llm-from-scratch/03_softmax_scaling.png' | relative_url }})
*The same five scores, with and without the $$1/\sqrt{d}$$ factor.*

## Two properties worth remembering

**Attention does not know about order.** If you shuffle the past tokens, each token gets the same weighted sum
(the scores only depend on the vectors, not on where they sit). That is exactly why Part 2 added position
embeddings to the token vectors before any attention happens.

**The mask is what makes the model causal.** The repository has a test for it: change a token at position 10 and
check that the predictions at positions 0 to 9 do not change at all.

```python
def test_causality():
    """Changing a future token must not change predictions for earlier positions."""
    model = tiny_model().eval()
    x = torch.randint(0, 40, (1, 16))
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % 40
    a, _ = model(x)
    b, _ = model(y)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-6)
    assert not torch.allclose(a[0, 10:], b[0, 10:])
```

Models *without* a mask exist too: encoders like BERT let every token see the whole sentence, which is great for
understanding text but cannot be used to generate it one token at a time.

## What a trained model attends to

Here are the attention weights of the four heads in the last layer of our trained model (Part 6) for the input
`"To be, or not to be"`. Row = the token doing the looking (query), column = the token being looked at (key).
Darker means more attention.

![Four 19 by 19 attention heatmaps for the text "To be, or not to be", one per head of the last layer.]({{ '/assets/img/llm-from-scratch/03_attention_heads.png' | relative_url }})
*Real attention patterns from our 0.8M-parameter model. Everything above the diagonal is the masked future.*

Even in this tiny model the four heads have clearly different jobs. Taking the strongest connection in each row:

- **Head 4 is local.** It looks at the current or the previous character almost every time (`e` → `b` in both
  `be`s, `r` → `o`). Knowing the last character or two is the single most useful thing for predicting the next.
- **Head 2 finds the start of the current word.** In `be`, `to` and `be` again, every letter attends to the space
  just before its word. That tells the model where in a word it is, which matters a lot for spelling.
- **Head 1 looks further back,** often at the last character of the *previous* word: both letters of the first
  `be` attend to the `o` of `To`, and the `n` and `o` of `not` attend to the `r` of `or`.
- **Head 3** is a mixture of local and word-level patterns.

Nobody programmed these roles. They emerged from training on next-character prediction, because each one makes
the next character easier to guess.

A word of caution: attention maps are suggestive, not proof. A head that "looks at" a token is not necessarily
*using* that information, and different heads interact. Interpreting what transformers compute is a research field
of its own (mechanistic interpretability).

## Try it yourself

> **Try it:** In `parts/03_attention.py`, change the input text for the heatmaps to a line with a character
> name, like `"ROMEO:\nO Romeo, Romeo"`. Do any heads look back at the earlier `Romeo`?

> **Exercise:** Remove the `masked_fill` line in the single-head example and print `weights`. What changes? Why
> would a model trained without the mask look great during training and fail when generating?

> **Exercise:** Our scores are $$q \cdot k$$. Why do we need *separate* query and key projections? What would
> go wrong if we used the same vector for both? (Hint: is $$q_i \cdot q_j$$ symmetric?)

## Key takeaways

- Attention is a **learned, data-dependent weighted average** over past tokens.
- Each token makes a query, a key and a value; scores are $$q \cdot k / \sqrt{d}$$, softmax turns them into
  weights, and the weights mix the values.
- The causal mask sets future scores to $$-\infty$$, so the model can never peek at the answer.
- Attention itself ignores order; position embeddings supply it.

**Next:** [Part 4: The transformer block]({{ '/blog/llm-from-scratch/part-4-transformer-block/' | relative_url }}),
where we run several heads in parallel and wrap them into a block we can stack.

### Further reading

- Vaswani et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).
- Bahdanau, Cho and Bengio (2014), [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473): where attention started.
- Jay Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/).
- Elhage et al. (2021), [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html), if you want to go deep on interpretability.
