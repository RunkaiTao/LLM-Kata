# Nano-GPT Kata

Build a GPT language model from scratch, one exercise at a time. Each exercise isolates a single concept — implement it in `exercise.py`, then run the tests to verify.

Based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

## Environment Setup

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch pytest
```

## How to Use

1. Work through the exercises in order (01 → 06, a → b → c within each section)
2. Open the `README.md` in each exercise folder — it explains the concept and what to implement
3. Fill in the functions/classes in `exercise.py` (look for `raise NotImplementedError`)
4. Run the tests to check your work:

```bash
cd 01_data_and_input/a_tokenizer
pytest test_exercise.py -v
```

## Exercises

### 01 Data and Input
- **a_tokenizer** — Character-level tokenization: build vocabulary, encode strings to token IDs, decode back
- **b_batch_loader** — Train/val split and random batch sampling with shifted targets

### 02 Layers
- **a_embedding** — Token and position embedding tables, summed element-wise
- **b_self_attention** — Single-head scaled dot-product attention with causal masking
- **c_multi_head_attention** — Multiple attention heads in parallel, concatenated and projected
- **d_feed_forward** — Position-wise feed-forward network (expand 4x, ReLU, compress)
- **e_transformer_block** — Pre-LayerNorm transformer block with residual connections

### 03 Combine Layers
- **a_assemble_model** — Wire all components into the full GPTLanguageModel, initialize weights

### 04 Output
- **a_forward_pass** — Complete forward pass from token indices to logits and cross-entropy loss

### 05 Train
- **a_loss_estimation** — Evaluate average loss on train/val splits with gradients disabled
- **b_training_loop** — AdamW optimizer, forward/backward pass, periodic evaluation

### 06 Inference
- **a_generate** — Autoregressive text generation with multinomial sampling
