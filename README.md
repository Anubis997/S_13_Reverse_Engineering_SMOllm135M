# Reverse Engineering SmolLM2-135M

This project recreates the SmolLM2-135M architecture and trains the synthesized model from scratch on generated text. The training run uses 5,000 steps, saves checkpoints every 500 steps, and continues for an additional 50 steps after the main run for final observation and evaluation.

The reference configuration is available in the SmolLM2 Nanotron checkpoint repository.

## Reference Architecture

The target SmolLM2-135M configuration has the following high-level characteristics:

| Component | Configuration |
|---|---|
| Parameters | ~135M |
| Attention heads | 9 |
| Key-value heads | 3 |
| Attention mechanism | Grouped Query Attention (GQA) |
| Activation function | SiLU / Swish |
| Vocabulary size | 49,152 |
| Maximum sequence length | 2,048 tokens |
| Original training corpus | Cosmopedia-v2 |

## Recreated Model

The replicated model closely matches the target parameter count and core architectural choices.

| Component | Recreated configuration |
|---|---|
| Parameters | 134.6M |
| Attention heads | 9 |
| Key-value heads | 3 |
| Tokenizer | cosmo2-tokenizer |
| Activation function | SwiGLU |
| Vocabulary size | 49,152 |
| Training sequence length | 750 tokens |

## Training Setup

The original SmolLM2 model was trained on Cosmopedia-v2, a corpus containing approximately 28 billion tokens. Training over that corpus is outside the scope of this project; the objective here is to understand model behavior, architecture replication, optimization dynamics, and generation quality over a limited 5,000-step run.

The intended setup used a sequence length of 2,048 tokens. However, the available Google Colab A100 environment supports a maximum practical sequence length of approximately 750 tokens for this experiment.

Reducing the context length from 2,048 to 750 tokens increases the number of steps required to expose the model to a similar total number of tokens:

```
2048 / 750 ≈ 2.73
```

Therefore, 5,000 steps at a sequence length of 2,048 correspond roughly to 13,650 steps at a sequence length of 750 when measured only by total token exposure. This is an approximation rather than an exact equivalence: shorter contexts cause more frequent truncation and reduce the model's ability to learn long-range dependencies.

For this project, training remains fixed at 5,000 steps with a 750-token context window to keep the experiment computationally practical.

## Checkpointing

Checkpoints are saved every 500 steps:

```
Checkpoint interval: 500 steps
Total training steps: 5,000
Number of main checkpoints: 10
Post-training continuation: 50 steps
```

Generation samples are collected at checkpoint intervals to observe changes in token prediction, syntax, dialogue formatting, and coherence over training.

## Generation Samples

The model was prompted with Shakespeare-style dialogue and asked to generate continuations. The outputs below show that the model begins to learn local formatting patterns—such as speaker labels, line breaks, and dramatic dialogue—but remains largely incoherent after 4,000–4,500 steps.

### Interval 4/10
**Average loss:** 0.0142

```
Input ending:
Forgiveness, horse! why do I rail on

Generated continuation:
thee
To be noble father died only nor it?

GLOUCESTER:
That whose easiest shall beheld!

KING RICHARD II:
We need not howl: let us bear it from heaven?
```

### Interval 5/10
**Average loss:** 0.0072

```
Generated continuation:
to cannot grievous into seest together,
Or if it till my natural king, and set forth
Are often damn no shadow.

GLOUCESTER:
Players, dead;, I but that stabbroke:
No, husband, it, are all down as good ladies,
```

### Interval 6/10
**Average loss:** 0.0052

```
Generated continuation:
IUS:
He is, sir.

CORIOLANUS:
Pray, I know how assured
As 'tis by proud man.
```

### Interval 7/10
**Average loss:** 0.0041

```
Generated continuation:
be cross it by.

GREMIO:
He straight shall any for over at hell burns?

PETRUCHIO:
Young cousin, my lord; I hence thou go.
```

### Interval 8/10
**Average loss:** 0.0035

```
Generated continuation:
k'd confound fellow; the duke was son, by
To tamed to wedded of doth he not,
By all the mean it of a prophetess
As vain: a and that answer it.
```

### Interval 9/10

```
Generated continuation:
and be
Of celebrationigh!

MENENIUS:
Let me, AUMBERLAND:
No, O, sorrow's no more as he till it.
```

## Observations

Training loss decreases consistently across the sampled checkpoints:

| Interval | Average loss |
|---|---|
| 4/10 | 0.0142 |
| 5/10 | 0.0072 |
| 6/10 | 0.0052 |
| 7/10 | 0.0041 |
| 8/10 | 0.0035 |

Despite the decline in loss, generated text remains mostly nonsensical. The model does show early evidence of learning structural features of the corpus:

- It produces speaker labels such as KING RICHARD II, GLOUCESTER, and MENENIUS
- It preserves dialogue-like formatting and line breaks
- It occasionally generates plausible short phrases
- It does not yet maintain grammatical consistency, semantic coherence, or character-level continuity

This gap between declining loss and weak generation quality is expected in an early-stage, from-scratch training experiment. Additional training steps, a broader training corpus, improved data quality, and longer context windows would likely be required before the model produces coherent continuations.
