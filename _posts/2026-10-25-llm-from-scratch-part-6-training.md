---
title: "Training: From Noise to Shakespeare"
subtitle: "Predict the next character, measure how wrong you were, nudge every weight a little. Repeat 3,000 times."
description: "Cross-entropy loss, next-token training batches, the training loop, AdamW, learning-rate warmup and cosine decay, gradient clipping, and reading a loss curve, with samples from every stage of training."
series: llm-from-scratch
series_title: LLMs from Scratch
part: 6
date: 2026-10-25 14:00:00 +0200
permalink: /blog/llm-from-scratch/part-6-training/
code: https://github.com/ilameya/llm-from-scratch/blob/main/tinyllm/train.py
image: /assets/img/llm-from-scratch/06_loss_curve.png
---

Our GPT from Part 5 is complete, and it writes complete nonsense: `'Cuw;suqW\n.&TTNUuuOgl...'`. Its 818,048
weights are random numbers. Training is the process that turns them into numbers that make Shakespeare likely.

The whole idea fits in one sentence: **show the model real text, measure how surprised it was by each next
character, and adjust every weight slightly so that it would be less surprised next time.** Let us make each part
of that sentence precise.

```bash
python parts/06_training.py      # trains the model used in Parts 2, 3 and 7
```

## How to measure "wrong": cross-entropy loss

For every position the model outputs a probability distribution over the 65 characters. Suppose the true next
character is `t`. If the model gave `t` probability $$p$$, we charge it

$$ \text{loss} = -\ln p $$

| probability the model gave the right answer | loss |
|---|---|
| 1.0 (certain and right) | 0.00 |
| 0.5 | 0.69 |
| 1/65 (pure guessing) | 4.17 |
| 0.001 (confident and wrong) | 6.91 |

Averaged over all positions this is the **cross-entropy loss**, the single number we train to minimise. It
punishes confident mistakes hard and rewards putting probability mass on the right answer. In code it is one line,
which we already saw in Part 5:

```python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

A handy way to read a loss value: $$e^{\text{loss}}$$ is the **perplexity**, roughly "how many characters the model
is choosing between". At the start, $$e^{4.17} = 65$$: all of them. A loss of 1.5 means about $$e^{1.5} \approx
4.5$$ plausible candidates per step.

## Where training examples come from

Language modelling needs no labels: the text *is* the label. We cut random windows out of the training text. The
target is the same window shifted one character to the left:

```python
def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu",
              generator: torch.Generator | None = None):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + block_size] for s in starts])
    return x.to(device), y.to(device)
```

![Input "First Ci" and target "irst Cit": each input position points to the target directly below it, which is the next character. At position 3 the model sees "Firs" and must predict "t".]({{ '/assets/img/llm-from-scratch/06_next_token_targets.svg' | relative_url }})
*One window of 8 characters is 8 training examples, thanks to the causal mask from Part 3.*

Here is a real window from the script:

```text
input  x: 'ou, trib'
target y: 'u, tribu'
  given 'o'          -> predict 'u'
  given 'ou'         -> predict ','
  given 'ou,'        -> predict ' '
  given 'ou, '       -> predict 't'
```

Because of the causal mask, position `t` can only see `x[0..t]`, so one forward pass trains all of these
predictions at once. With `batch_size = 32` and `block_size = 128`, every step trains on 4,096 next-character
predictions.

**Train and validation split.** We keep the last 10% of the text aside and never train on it. Loss on this
*validation* set tells us how well the model does on text it has not seen, which is what we actually care about.

## The training loop

![One training step: sample a batch, forward pass, loss, backward pass, update weights, then repeat.]({{ '/assets/img/llm-from-scratch/06_training_loop.svg' | relative_url }})
*Every deep learning training loop has these five steps.*

Here is the heart of `tinyllm/train.py`:

```python
for group in opt.param_groups:      # 1. set the learning rate for this step
    group["lr"] = lr_at(step, tc)
x, y = get_batch(train_data, model.cfg.block_size, tc.batch_size, device)
_, loss = model(x, y)               # 2. forward pass: how wrong are we?
opt.zero_grad(set_to_none=True)
loss.backward()                     # 3. backward pass: gradients of the loss
torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)  # 4. avoid huge steps
opt.step()                          # 5. update the weights
```

`loss.backward()` is where PyTorch's autograd does its magic: it computes, for every one of the 818,048 weights,
how the loss would change if that weight changed slightly (the **gradient**). The optimiser then moves each weight
a small step in the direction that lowers the loss.

### The optimiser: AdamW

Plain gradient descent moves every weight by `learning_rate × gradient`. **Adam** improves on this by keeping a
running average of each weight's gradients (momentum, to smooth out noise) and of their squared size (so weights
with consistently small gradients still move). **AdamW** adds *weight decay*: a gentle pull of every weight towards
zero, which discourages the model from relying on huge weights and helps it generalise.

```python
def configure_optimizer(self, lr: float, weight_decay: float = 0.1, betas=(0.9, 0.95)):
    """AdamW with weight decay on matrices only (not on biases / LayerNorm gains)."""
    params = [p for p in self.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)
```

We only decay the weight *matrices* (embeddings and linear layers). Biases and LayerNorm scales are vectors with
special jobs, and pulling them towards zero only hurts.

### The learning-rate schedule

The learning rate is the size of each step. Too large and training explodes; too small and it crawls. We change it
during training:

```python
def lr_at(step: int, tc: TrainConfig) -> float:
    """Linear warmup, then cosine decay from lr down to min_lr."""
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    progress = (step - tc.warmup_steps) / max(1, tc.steps - tc.warmup_steps)
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))
```

![The learning rate rises linearly over the first 100 steps to 0.001 and then follows a cosine curve down to 0.0001 at step 3000.]({{ '/assets/img/llm-from-scratch/06_lr_schedule.png' | relative_url }})
*Start gently, go fast, finish gently.*

- **Warmup.** At the start Adam's running averages are empty and gradients are large and erratic. A small learning
  rate for the first 100 steps avoids a bad jump that can ruin the run.
- **Cosine decay.** Late in training we want to settle into a good minimum rather than bounce around it, so the
  step size shrinks smoothly to a tenth of its peak.

### Gradient clipping

Occasionally one batch produces an unusually large gradient. `clip_grad_norm_` rescales the whole gradient
whenever its overall length exceeds 1.0, so a single odd batch can never throw the model far off course.

## Watching it learn

Here is the full run with the default settings (3,000 steps, 818,048 parameters), on a CPU:

```text
step     0 | train 4.187 | val 4.189 | lr 1.00e-05 |    2.0s
step   250 | train 2.375 | val 2.397 | lr 9.94e-04 |   71.7s
step   500 | train 2.157 | val 2.200 | lr 9.58e-04 |  211.8s
step  1000 | train 1.834 | val 1.974 | lr 8.03e-04 |  358.3s
step  2000 | train 1.578 | val 1.757 | lr 3.39e-04 |  646.1s
step  3000 | train 1.499 | val 1.687 | lr 1.00e-04 |  934.6s
```

The full run took 16 minutes on a modest 2-core cloud CPU; most laptops are faster. Validation loss
falls from 4.19 to 1.69, which is a perplexity of $$e^{1.69} \approx 5.4$$: the model has narrowed 65 possible characters
down to about 5 plausible ones at each step.

![Training and validation loss falling from 4.2 to 1.69 over 3,000 steps, with the noisy per-batch loss in the background and a dotted line at ln(65) = 4.17 for random guessing.]({{ '/assets/img/llm-from-scratch/06_loss_curve.png' | relative_url }})
*Gray: loss on each individual batch. Blue and orange: averages over 20 batches, on training and validation text.*

The numbers are nice, but the samples are better. The script generates text from the prompt `ROMEO:` at four
moments during training (same random seed each time):

**Step 0 (random weights):**

```text
ROMEO:t3ZrjmhfPqkARxLsXltUWRjHcu
?exhAAU&&NdkIxAlsv,BUMyccRu$BygXOfntw
kVrM z&?j DjRbsOEHelC!BTg3ee&PPCslMp--3KyporWiXkMvRJIRc
```

**Step 250:**

```text
ROMEO:
Thy mhis ke whenlle louce mer,
An, id I tle fend cofutoy he ntherere he?

Sou s ENERT:
Theserd tst atho y we wak we ath
```

**Step 1,000:**

```text
ROMEO:
Threm far her sellf, our aner,
And in I to the My of hey his thee
re hath hou sherelf Beaned in stakes, your we peer
Re
```

**Step 3,000 (end of training):**

```text
ROMEO:
Why more know self it one means.

PAULINA:
And My pourty his treath? hear think the seem.

BRUTUS:
We'll percious a man
```

You can watch the model discover, in order: which characters are common, that text comes in words separated by
spaces, that lines are short and character names are in capitals followed by a colon, and finally real English
words and something like the rhythm of verse. Nobody programmed any of these rules. They are all just ways of
making the next character less surprising.

## Reading a loss curve

A few things to look for in your own runs:

- **Train and validation together going down:** the model is learning things that generalise. 
- **A growing gap between them:** the model starts memorising the training text (overfitting). Ours ends with a
  gap of 0.19 (1.50 train vs. 1.69 validation). More dropout, more data or a smaller model would reduce it.
- **Loss stuck near 4.17:** the model learns nothing. Check the learning rate and that `targets` really are the
  inputs shifted by one.
- **Loss suddenly jumping up or becoming `nan`:** the learning rate is too high, or gradient clipping is off.

Two debugging habits that save hours, both in our test suite: check that the **initial loss is close to
$$\ln V$$** (Part 5), and check that the model can **overfit a single batch** to nearly zero loss. If it cannot
memorise four short sequences, something in the model or the loop is broken.

```python
def test_can_overfit_one_batch():
    model = tiny_model()
    x = torch.randint(0, 40, (4, 16))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2
```

## Try it yourself

> **Try it:** Train with `--lr 1e-2` and with `--lr 1e-4` for 1,000 steps each. Plot the three loss curves
> (the history is saved next to the checkpoint as JSON). What happens in each case?

> **Exercise:** Set `--dropout 0.0` and train for 6,000 steps. Does the gap between train and validation loss
> grow? At which step does validation loss stop improving?

> **Exercise:** Train the BPE version (`--tokenizer bpe`). Its loss is *per token*, and each token is about two
> characters. How would you compare its loss fairly with the character model's? (Hint: convert to loss per
> character.)

## Key takeaways

- The training objective is **cross-entropy**: $$-\ln p$$ of the true next token, averaged over every position.
- Every window of text is many training examples at once; the targets are the inputs shifted by one.
- The loop: batch → forward → loss → backward → optimiser step, with AdamW, warmup + cosine decay and gradient
  clipping.
- Watch train and validation loss together; their gap tells you about overfitting.

**Next:** [Part 7: Generating text]({{ '/blog/llm-from-scratch/part-7-sampling/' | relative_url }}), where we look
at how to turn the model's probabilities into text, and why that choice matters as much as the training.

### Further reading

- Kingma and Ba (2014), [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980).
- Loshchilov and Hutter (2017), [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101) (AdamW).
- Andrej Karpathy, [*A Recipe for Training Neural Networks*](https://karpathy.github.io/2019/04/25/recipe/): practical debugging advice.
