---
title: "Embeddings: Giving Tokens Meaning and Position"
subtitle: "Token ids are just labels. Embeddings turn them into vectors the network can learn from, and add a sense of where each token sits."
description: "Token embeddings as learnable lookup tables, why one-hot vectors are not enough, why position matters, learned vs. sinusoidal positional embeddings, and what a trained model's embeddings look like."
series: llm-from-scratch
series_title: LLMs from Scratch
part: 2
date: 2026-10-25 10:00:00 +0200
permalink: /blog/llm-from-scratch/part-2-embeddings/
code: https://github.com/ilameya/llm-from-scratch/blob/main/parts/02_embeddings.py
image: /assets/img/llm-from-scratch/02_embeddings.png
---

At the end of Part 1 our text had become a list of integers: `"To be"` is `[32, 53, 1, 40, 43]`. These numbers are
**labels**, not quantities. Token 53 (`o`) is not "bigger" than token 32 (`T`), and 40 and 43 being close says
nothing about `b` and `e`. If we fed the raw ids into a network, it would try to find meaning in their size.

What we want instead is a representation where every token is described by **many numbers**, and where those
numbers are *learned* so that tokens which behave similarly end up with similar descriptions. That representation
is an **embedding**.

```bash
python parts/02_embeddings.py
```

## Attempt 1: one-hot vectors

A classic fix for labels: represent token `i` as a vector of zeros with a single 1 at position `i`. With 65
tokens, every token becomes a 65-dimensional **one-hot** vector.

That removes the fake ordering, but has two problems. Every pair of tokens is equally far apart (the vectors are
orthogonal), so `a` is as different from `e` as from `?`. And the size grows with the vocabulary: with GPT-2's
50,257 tokens, each vector would have 50,257 entries, almost all zero.

## Attempt 2: a learned lookup table

An **embedding table** is a matrix with one row per token, of shape `(vocab_size, n_embd)`. To embed token `i`, we
simply take row `i`. The rows start random and are trained along with everything else.

```python
vocab_size, n_embd = 10, 4
emb = nn.Embedding(vocab_size, n_embd)
```

```text
embedding table shape: (10, 4)
emb(3)            : [[ 0.12   1.238  1.117 -0.247]]
one_hot(3) @ table: [[ 0.12   1.238  1.117 -0.247]]
-> looking up row 3 == multiplying a one-hot vector by the table
```

The two results are identical. An embedding *is* a one-hot vector multiplied by a weight matrix, implemented as
an efficient lookup. This tells us how it learns: like any linear layer, through gradients. When the model makes a
mistake involving token 3, backpropagation nudges row 3 of the table.

The width `n_embd` (we use 128, GPT-2 small uses 768) is a design choice: each token gets `n_embd` numbers to
describe itself. Nobody tells the model what these numbers should mean. It discovers whatever features help it
predict the next token: is this a vowel? uppercase? punctuation that ends a line?

## Position matters

Here is a subtle but important problem. In Part 3 we will build attention, which combines token vectors with
weighted sums. A sum does not care about order. We can see the issue with a toy example:

```text
sum of embeddings, 'dog bites man': [ 0.045 -1.724 -0.217 -2.241]
sum of embeddings, 'man bites dog': [ 0.045 -1.724 -0.217 -2.241]
identical? True -> without position information, order is invisible
```

Same words, opposite meanings, identical representation. For characters it is even worse: `"no"` and `"on"`
would look the same. The model needs to know **where** each token is.

## Position embeddings

The GPT-2 solution is pleasingly simple: a second embedding table, with one row per **position** instead of per
token. Position 0 gets row 0, position 1 gets row 1, and so on, up to the maximum context length `block_size`.
The input to the transformer is the **sum** of the two:

```python
pos = torch.arange(start, start + T, device=idx.device)          # (T,)
x = self.tok_emb(idx) + self.pos_emb(pos)                         # (B, T, C)
```

![Each of the five tokens T, o, space, b, e has an id; the id selects a row of the token embedding table and the position selects a row of the position embedding table; the two vectors are added to give the model's input vector for that position.]({{ '/assets/img/llm-from-scratch/02_embeddings.svg' | relative_url }})
*The input to the model: "what" plus "where", for every position.*

The result `x` has shape `(B, T, C)`: for every sequence in the batch, for every position, one `C`-dimensional
vector. This shape stays the same through the entire transformer (Parts 3 to 5).

**Why add instead of concatenating?** Concatenating would double the width of everything downstream. Adding keeps
the width fixed, and in 128 dimensions there is plenty of room: the model can learn to keep token information and
position information in different directions of the same vector. It works well in practice, and every GPT model
does it this way.

One consequence: because the table has exactly `block_size` rows, our model **cannot process sequences longer
than `block_size`** (128 characters for us). There is simply no row for position 128.

## The alternative: sinusoidal positions

The original Transformer paper did not learn position vectors. It computed them with sine and cosine waves of
different frequencies:

$$ PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/C}}\right), \qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/C}}\right) $$

![Heatmap of sinusoidal positional encodings: 128 positions on the horizontal axis and 64 dimensions on the vertical axis. Low dimensions oscillate quickly, higher dimensions slowly.]({{ '/assets/img/llm-from-scratch/02_sinusoidal_pe.png' | relative_url }})
*Low dimensions oscillate fast, high dimensions slowly, like the hands of a clock. Every position gets a unique pattern.*

The low dimensions change quickly from one position to the next and the high dimensions slowly, a bit like the
second, minute and hour hands of a clock. Together they give every position a unique fingerprint without any
training. Both approaches work about equally well at small scale; GPT-2 chose learned embeddings for simplicity.
Most modern models use a third option, RoPE, which we will meet in Part 8.

## What did our model learn?

After training (Part 6) we can open up the embedding tables. Each character's embedding is a 128-dimensional
vector, too many to draw, so we project them to 2D with PCA (principal component analysis), which keeps the two
directions along which the embeddings vary most.

![Scatter plot of the 65 character embeddings projected to two dimensions, coloured by vowel, lowercase consonant, uppercase and space or punctuation.]({{ '/assets/img/llm-from-scratch/02_learned_token_embeddings.png' | relative_url }})
*Nobody told the model which characters are vowels. Colours are added by us, after training.*

The model was never told what a vowel or a capital letter is, yet the embedding has sorted the alphabet:

- **Uppercase letters** form their own cluster at the top. Capitals behave differently from lowercase letters in
  this text: they start lines and character names (`ROMEO:`), so similar things tend to follow them.
- **Lowercase vowels** (`a e i o u`) sit together, next to but separate from the lowercase consonants.
- **Punctuation** that ends or splits a clause (`, . ; ! ?`) sits in the middle, and the newline `\n` is off on its own.
- The far right holds the **rare characters**: `&` appears 3 times in the whole dataset and `$` once, followed by
  `X Z Q J x q z j` and the digit `3`. With so few examples their embeddings barely moved from their random start,
  which is exactly the "rare tokens learn little" problem from Part 1.

This is the core promise of embeddings: tokens that play similar roles end up with similar vectors, purely from
the pressure to predict the next character.

The position table tells a story too. For every pair of positions we compute the cosine similarity of their
learned vectors:

![Heatmap of cosine similarity between the 128 learned position embeddings.]({{ '/assets/img/llm-from-scratch/02_learned_position_similarity.png' | relative_url }})
*Cosine similarity between learned position vectors.*

A bright band runs along the diagonal: every position is most similar to its neighbours, and the similarity fades
smoothly over a distance of about 20 to 30 positions. Positions far apart are almost orthogonal. Again nobody
asked for this. The table started as 128 independent random vectors, and the model *learned* that position 41
should look like 40 and 42, because nearby positions play similar roles in a sentence. Its learned table ended up
with the same smooth structure that the sinusoidal encoding builds in by design.

## Try it yourself

> **Try it:** In `parts/02_embeddings.py`, find the five nearest neighbours of `'a'` and of `'.'` in the trained
> token embedding (use cosine similarity). Do they make sense?

> **Exercise:** Train a model *without* position embeddings (replace `self.pos_emb(pos)` with `0`). How much
> worse is the validation loss? Look at samples: what goes wrong?

> **Exercise:** Swap the learned position table for the sinusoidal one (computed once, not trained). Does the
> loss change? Can the sinusoidal version handle positions beyond 128?

## Key takeaways

- Token ids are labels. An **embedding table** maps each id to a learned vector (one row per token).
- An embedding lookup equals a one-hot vector times a weight matrix, so it is trained by backpropagation like any
  layer.
- Attention ignores order, so we **add a position embedding** to each token embedding.
- Learned position tables (GPT-2) limit the context to `block_size`; sinusoidal and rotary encodings are the
  alternatives.

**Next:** [Part 3: Self-attention]({{ '/blog/llm-from-scratch/part-3-attention/' | relative_url }}), where these
vectors finally start talking to each other.

### Further reading

- Mikolov et al. (2013), [*Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/abs/1301.3781) (word2vec): the paper that made learned embeddings famous.
- Vaswani et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), section 3.5 on positional encodings.
- Amirhossein Kazemnejad, [*Transformer Architecture: The Positional Encoding*](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/): a gentle visual explanation.
