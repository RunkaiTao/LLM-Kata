"""
Full GPT-2 model assembly with ModuleDict, weight tying, and scaled initialization.

This exercise brings together all the building blocks into the complete GPT-2
architecture. Key differences from the simpler nano-gpt kata:

1. nn.ModuleDict (not nn.Sequential) — stores wte, wpe, h, ln_f as named sub-modules
2. Weight tying — the token embedding (wte) shares weights with the output projection (lm_head)
3. Scaled initialization — layers with NANOGPT_SCALE_INIT get smaller initial weights
   to prevent the residual stream variance from growing with depth

Reference: Karpathy's build-nanogpt train_gpt2.py lines 79-128
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Import completed exercises: GPTConfig, Block
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig
Block = load("01_model_architecture", "d_transformer_block").Block


# ---------------------------------------------------------------------------
# YOUR TASK: Implement GPT.__init__, _init_weights, and forward
# ---------------------------------------------------------------------------
class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        """
        Args:
            config: GPTConfig instance.

        Steps:
        1. Store config as an instance attribute
        2. Create self.transformer as an nn.ModuleDict with named sub-modules:
           - wte: token embedding (vocab_size -> n_embd) (use nn.Embedding)
           - wpe: position embedding (block_size -> n_embd) (use nn.Embedding)
           - h: list of n_layer transformer blocks (use nn.ModuleList with Block)
           - ln_f: final layer norm over n_embd (use nn.LayerNorm)
        3. Create self.lm_head — linear output projection from n_embd to vocab_size,
           no bias (use nn.Linear)
        4. Tie wte and lm_head weights so they share the same parameter
        5. Apply weight initialization to all sub-modules (use self.apply with _init_weights)
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        # Step 1: self.config = ...
        # Step 2: self.transformer = nn.ModuleDict(dict(
        #             wte = ...   (nn.Embedding: vocab_size -> n_embd)
        #             wpe = ...   (nn.Embedding: block_size -> n_embd)
        #             h = ...     (nn.ModuleList of n_layer Block instances)
        #             ln_f = ...  (nn.LayerNorm over n_embd)
        #         ))
        # Step 3: self.lm_head = ...  (nn.Linear: n_embd -> vocab_size, no bias)
        # Step 4: self.transformer.wte.weight = self.lm_head.weight  (weight tying)
        # Step 5: self.apply(self._init_weights)
        pass

    def _init_weights(self, module):
        """
        Initialize weights for the model.

        Rules:
        - For nn.Linear modules:
            Initialize weights from a normal distribution with std=0.02
            If the module has the NANOGPT_SCALE_INIT flag, scale std down
            by (2 * n_layer)^(-0.5) to account for residual stream accumulation
            Zero-initialize bias if present
        - For nn.Embedding modules:
            Initialize weights from a normal distribution with std=0.02

        (use torch.nn.init.normal_ and torch.nn.init.zeros_, check hasattr for the flag)
        """
        # TODO: Implement _init_weights following the rules above
        # if isinstance(module, nn.Linear):
        #     std = ...  (base std=0.02, scale by (2 * n_layer)**-0.5 if NANOGPT_SCALE_INIT)
        #     torch.nn.init.normal_(module.weight, ...)
        #     if module.bias is not None: torch.nn.init.zeros_(module.bias)
        # elif isinstance(module, nn.Embedding):
        #     torch.nn.init.normal_(module.weight, ...)
        pass

    def forward(self, idx, targets=None):
        """
        Args:
            idx: Token indices of shape (B, T).
            targets: Target token indices of shape (B, T), or None.

        Returns:
            logits: Tensor of shape (B, T, vocab_size).
            loss: Scalar cross-entropy loss, or None if no targets.

        Steps:
        1. Unpack B, T from idx and assert T does not exceed block_size
        2. Create position indices 0..T-1 on the same device as idx (use torch.arange)
        3. Look up token embeddings (wte) and position embeddings (wpe)
        4. Add token and position embeddings (position broadcasts across batch)
        5. Pass through each transformer block in self.transformer.h
        6. Apply final layer norm (ln_f)
        7. Project to vocabulary logits via lm_head -> (B, T, vocab_size)
        8. If targets provided, compute cross-entropy loss after flattening
           logits and targets (use F.cross_entropy); otherwise loss is None
        9. Return (logits, loss)
        """
        # TODO: Implement forward following the steps above
        # Step 1: B, T = ...  (unpack from idx, assert T <= block_size)
        # Step 2: pos = ...     (torch.arange 0..T-1, same device as idx)
        # Step 3: pos_emb = ... (position embeddings via wpe)
        #         tok_emb = ... (token embeddings via wte)
        # Step 4: x = ...      (tok_emb + pos_emb)
        # Step 5: for block in self.transformer.h: x = block(x)
        # Step 6: x = ...      (apply ln_f)
        # Step 7: logits = ... (project through lm_head)
        # Step 8: loss = None  (if targets: compute F.cross_entropy on flattened logits/targets)
        # return logits, loss
        pass

# Run tests: pytest nano-gpt2/01_model_architecture/e_gpt_model/test_exercise.py -v
# Test individual parts:
# pytest nano-gpt2/01_model_architecture/e_gpt_model/test_exercise.py -v -k TestModuleDict
# pytest nano-gpt2/01_model_architecture/e_gpt_model/test_exercise.py -v -k TestWeightTying
# pytest nano-gpt2/01_model_architecture/e_gpt_model/test_exercise.py -v -k TestScaledInit
# pytest nano-gpt2/01_model_architecture/e_gpt_model/test_exercise.py -v -k TestForwardPass
