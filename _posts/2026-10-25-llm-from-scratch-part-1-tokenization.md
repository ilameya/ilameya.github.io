---
title: "From Text to Numbers: Tokenization"
subtitle: "A neural network can only multiply numbers. So the first job is turning text into numbers, and back."
description: "What a language model actually does, and how character-level and Byte Pair Encoding (BPE) tokenizers turn text into token ids."
series: llm-from-scratch
series_title: LLMs from Scratch
part: 1
date: 2026-10-25 09:00:00 +0200
permalink: /blog/llm-from-scratch/part-1-tokenization/
code: https://github.com/ilameya/llm-from-scratch/blob/main/parts/01_tokenization.py
image: /assets/img/llm-from-scratch/01_big_picture.png
---

Welcome to **LLMs from Scratch**, a series where we build a small GPT, piece by piece, and train it on
Shakespeare on a normal laptop. Every line of code is in the
[companion repository](https://github.com/ilameya/llm-from-scratch), and every part comes with a script you can run.

By the end of the series you will be able to explain, and implement, everything between the text you type and
the text a language model writes back. No GPU, no magic, and under 700 lines of heavily commented Python.

## What a language model actually does

Strip away the chat interface and a large language model does exactly one thing:

> Given some text, it outputs a probability for every possible *next* piece of text.

That is all. Writing a whole paragraph is that one trick on repeat: predict a probability distribution, pick a
token from it, append the token, predict again.

![The big picture: text goes through a tokenizer, becomes token ids, flows through the GPT model, which outputs next-token probabilities. A token is sampled, appended to the text, and the loop repeats.]({{ '/assets/img/llm-from-scratch/01_big_picture.svg' | relative_url }})
*The whole series on one page. Today: the tokenizer.*

Here is the roadmap:

| Part | Topic | The question it answers |
|---|---|---|
| 1 | Tokenization | How does text become numbers? |
| 2 | Embeddings | How do numbers become vectors with meaning and position? |
| 3 | Self-attention | How do tokens share information? |
| 4 | The transformer block | How do we stack that into something deep? |
| 5 | Assembling GPT | What does the complete model look like? |
| 6 | Training | How does it learn from text? |
| 7 | Sampling | How do we turn probabilities into writing? |
| 8 | Scaling | What changes between our toy and ChatGPT? |

## Setup

```bash
git clone https://github.com/ilameya/llm-from-scratch.git
cd llm-from-scratch
pip install -r requirements.txt
python parts/01_tokenization.py
```

Our dataset is **Tiny Shakespeare**: about 1.1 million characters of Shakespeare's plays in one text file.
It is small enough to train on in minutes and structured enough that we will clearly see the model learn
(character names, line breaks, old English spelling).

```text
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.
```

## Why we need a tokenizer

A neural network is a big function made of matrix multiplications. It takes numbers in and gives numbers out.
So we need two functions:

- `encode(text) -> list of integers`
- `decode(list of integers) -> text`

with `decode(encode(text)) == text`. Each integer is a **token id**, and the set of all possible tokens is the
**vocabulary**. That is a tokenizer. The interesting question is: *what should one token be?*

## Option 1: one token per character

The simplest answer: every distinct character is a token. Shakespeare uses 65 different characters, so our
vocabulary has 65 entries.

```python
class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # string -> int
        self.itos = {i: ch for i, ch in enumerate(chars)}  # int -> string

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)
```

Running it:

```text
[char] vocabulary size: 65
[char] vocabulary: "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
[char] encode('Hello there!') -> [20, 43, 50, 50, 53, 1, 58, 46, 43, 56, 43, 2]
[char] decode(...)            -> 'Hello there!'
```

Notice that both `l`s in "Hello" become `50`. The ids are just positions in the sorted list of characters; the
number 50 carries no meaning by itself. Giving tokens meaning is the job of Part 2.

Character tokenizers are wonderful for learning: tiny vocabulary, no training, nothing hidden. But they have
two problems:

1. **Sequences are long.** Every character is one step for the model. The model can only look at a fixed number
   of tokens at a time (its *context window*), so long sequences mean it sees less text.
2. **They do not generalise.** Our tokenizer has never seen `é` or an emoji. `encode("café")` crashes.

## Option 2: bytes

Every string a computer stores is already a list of numbers: its UTF-8 bytes, each between 0 and 255.

```text
'café 🙂' has 6 characters but 10 UTF-8 bytes: [99, 97, 102, 195, 169, 32, 240, 159, 153, 130]
```

With bytes, the vocabulary is fixed at 256 and *any* text can be encoded. Problem 2 solved. But the sequences
got even longer (`é` is two bytes, the emoji is four). Can we keep the "works on anything" property but make
sequences shorter?

## Option 3: Byte Pair Encoding (BPE)

BPE, the tokenizer family used by GPT-2, GPT-4, Llama and most modern models, starts from bytes and then
**learns** new tokens from data with one simple rule:

> Find the pair of neighbouring tokens that appears most often. Merge it into a new token. Repeat.

![Byte Pair Encoding on the words the, thin and then. At the start every letter is a token; the pair t,h is most frequent (three times) and becomes th. Then th,e is most frequent (twice) and becomes the. Eleven tokens become six.]({{ '/assets/img/llm-from-scratch/01_bpe_merges.svg' | relative_url }})
*Two merges turn 11 tokens into 6. Frequent chunks of text become single tokens.*

The whole algorithm needs two helpers. One counts neighbouring pairs:

```python
def get_pair_counts(ids: list[int], counts: Counter | None = None, weight: int = 1) -> Counter:
    """Count how often each adjacent pair appears. [1,2,3,1,2] -> {(1,2):2, (2,3):1, (3,1):1}"""
    counts = Counter() if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
    return counts
```

The other replaces every occurrence of a pair with a new id:

```python
def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of ``pair`` in ``ids`` with ``new_id``."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
```

Training is a loop: count, pick the most frequent pair, merge, remember the merge.

```python
for new_id in range(256, vocab_size):
    stats: Counter = Counter()
    for c, ids in chunks.items():
        get_pair_counts(ids, stats, weight=chunk_counts[c])
    pair = max(stats, key=stats.get)          # most frequent pair
    for c in chunks:
        chunks[c] = merge(chunks[c], pair, new_id)
    self.merges[pair] = new_id
    self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
```

Two details worth knowing:

- **Chunks.** Before training we split the text into word-like chunks with a regular expression (an optional
  space followed by letters, digits or punctuation). Merges never cross a chunk boundary, so we never learn
  silly tokens like `"e. T"`. GPT-2 does the same with a more elaborate regex. Counting each *unique* chunk once
  (with its frequency as a weight) also makes training much faster.
- **Encoding new text** applies the learned merges in the order they were learned: always the earliest merge
  first, exactly as during training.

Here are the first merges our tokenizer learns from Shakespeare:

```text
merge    1: (32, 116) -> 256 ' t' (seen 23835x)
merge    2: (104, 101) -> 257 'he' (seen 18203x)
merge    3: (32, 97) -> 258 ' a' (seen 13541x)
merge    4: (111, 117) -> 259 'ou' (seen 12730x)
...
merge   12: (256, 257) -> 267 ' the' (seen 8683x)
```

Merge 12 combines two *learned* tokens (`' t'` and `'he'`) into `' the'`. That is how BPE builds longer and
longer tokens: merges of merges. After 256 merges (a vocabulary of 512) the longest tokens are whole words:

```text
[bpe] longest tokens: [' shall', ' there', ' would', ' that', ' with', ' your', ' thou', ' have', ' this', ' will']
```

And a famous line now looks like this (`|` separates tokens):

```text
[bpe] 'ROMEO: But, soft! what light through yonder window breaks?'
[bpe] -> 32 tokens: R|O|M|E|O|:| |But|,| so|f|t|!| what| li|ght| th|r|ou|gh| y|ond|er| w|ind|ow| b|re|a|k|s|?
[char] -> 58 tokens
```

Common words (`' what'`, `'But'`) are single tokens. Rare ones (`yonder`) are built from pieces. Capitalised
names stay as single letters, because in only 256 merges uppercase pairs never became frequent enough.

## The trade-off: vocabulary size vs. sequence length

More merges give longer tokens, so the same text needs fewer of them:

![Characters per token rises from 1.0 at 256 tokens to 1.51 at 320, 1.96 at 512 and 2.44 at 1024.]({{ '/assets/img/llm-from-scratch/01_bpe_compression.png' | relative_url }})
*Measured on the first 100,000 characters of Tiny Shakespeare.*

So why not use a million tokens? Because every token needs its own row in the embedding table and its own
output score (we will see both in Parts 2 and 5). A bigger vocabulary means a bigger model, and rare tokens are
seen so seldom during training that the model learns little about them. Real models settle around 32,000 to
200,000 tokens, trained on far more text than ours.

> **Why LLMs are bad at counting letters.** A model that sees `' strawberry'` as two or three tokens never
> directly "sees" the individual letters `r`. Many famous LLM quirks (spelling, reversing words, arithmetic on
> long numbers) come from tokenization, not from the network.

## Which tokenizer will we use?

For the rest of the series we use the **character tokenizer**. With only 65 tokens our model stays tiny, training
is fast, and when we look at attention patterns (Part 3) each row is one readable character. The repository
supports both, so you can switch any time:

```bash
python -m tinyllm.train --tokenizer bpe --bpe_vocab 512
```

## Try it yourself

> **Try it:** Train a BPE tokenizer with `verbose=True` on a different text (your own notes, a Python file, a
> text in another language). Which merges come first? What does that tell you about the text?

> **Exercise:** `BPETokenizer.encode` handles text it never saw during training, including emoji. Why can it
> never fail, while `CharTokenizer.encode` can? Check your answer with `tok.encode("Héllo 🙂")`.

> **Exercise:** Our encoder recomputes all pairs after each merge, which is slow for long texts. Can you think of
> a faster way? (Hint: real tokenizers like `tiktoken` are written in Rust and use a priority queue.)

## Key takeaways

- A language model predicts **one token at a time**; everything else is a loop around that.
- A tokenizer maps text to integer ids and back. The ids themselves carry no meaning.
- Character tokenizers are simple but give long sequences; byte-level BPE works on any text and learns frequent
  chunks as single tokens.
- Vocabulary size trades sequence length against model size.

**Next:** [Part 2: Embeddings]({{ '/blog/llm-from-scratch/part-2-embeddings/' | relative_url }}), where integer ids
become vectors that the network can actually learn from.

### Further reading

- Sennrich, Haddow and Birch (2016), [*Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909): the paper that brought BPE to NLP.
- Radford et al. (2019), [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): the GPT-2 paper, which introduced byte-level BPE.
- Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe): a slightly more complete BPE implementation, also written to be read.
