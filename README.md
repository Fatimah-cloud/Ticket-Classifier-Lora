# Support Ticket Classifier LoRA Fine-Tuning

Fine-tuning Qwen2.5-1.5B-Instruct with LoRA to reliably classify support tickets into 4 categories: `Billing`, `Technical`, `Account`, `General`.


## What this is

A small, from-scratch example of parameter-efficient fine-tuning (PEFT): instead of updating all 1.5B parameters of the base model,LoRA freezes the pretrained weights and trains a small set of low-rank adapter matrices injected into the attention layers — about 4.4M trainable parameters (0.28%) of the total model.

The goal isn't to teach the model new facts it's to teach it a consistent output format and exact label vocabulary, which is exactly the kind of narrow behavioral shift LoRA is well suited for.

## Results

| Evaluation | Baseline (untrained) | Fine-tuned |
|---|---|---|
| Matched prompt (same format as training) | 18/30 (60%) | **30/30 (100%)** |
| No-hint prompt (category list withheld) | lower / inconsistent | improved formatting, partial label convergence |

The base model frequently invented its own category names (e.g. `"Suspension"`, `"RateLimit"`) or wrapped answers in extra text/quotes. After fine-tuning, the model reliably outputs one of the exact 4 trained category names when prompted the way it was trained — and shows partial (imperfect) improvement even under a stricter, out-of-distribution prompt it never saw during training.


## Data

A ~200-example hand-curated synthetic dataset of support tickets, 50 per category, 85/15 train/test split.


## limitations

- The dataset is intentionally small and synthetic; a larger, more diverse dataset (and/or more training epochs) would likely close the gap seen in the no-hint generalization test.
