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
        1. Store self.config = config
        2. Create self.transformer = nn.ModuleDict containing:
           - wte = nn.Embedding(config.vocab_size, config.n_embd)  — token embeddings
           - wpe = nn.Embedding(config.block_size, config.n_embd)  — position embeddings
           - h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
           - ln_f = nn.LayerNorm(config.n_embd)  — final layer norm
        3. Create self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        4. Weight tying: self.transformer.wte.weight = self.lm_head.weight
           — this makes the token embedding and output projection share the same parameters
        5. Apply weight initialization: self.apply(self._init_weights)
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight
        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for the model.

        Rules:
        - If module is nn.Linear:
            std = 0.02
            If module has attribute NANOGPT_SCALE_INIT:
                std *= (2 * self.config.n_layer) ** -0.5
            Set weights to Normal(mean=0.0, std=std)
            If bias exists, set to zeros.
        - If module is nn.Embedding:
            Set weights to Normal(mean=0.0, std=0.02)

        Hint: Use torch.nn.init.normal_() and torch.nn.init.zeros_()
        """
        # TODO: Implement _init_weights following the rules above
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Args:
            idx: Token indices of shape (B, T).
            targets: Target token indices of shape (B, T), or None.

        Returns:
            logits: Tensor of shape (B, T, vocab_size).
            loss: Scalar cross-entropy loss, or None if no targets.

        Steps:
        1. B, T = idx.size()
        2. Assert T <= self.config.block_size
        3. pos = torch.arange(0, T, dtype=torch.long, device=idx.device) -> (T,)
        4. pos_emb = self.transformer.wpe(pos)  -> (T, n_embd)
        5. tok_emb = self.transformer.wte(idx)  -> (B, T, n_embd)
        6. x = tok_emb + pos_emb  (pos_emb broadcasts across batch dimension)
        7. For each block in self.transformer.h: x = block(x)
        8. x = self.transformer.ln_f(x)
        9. logits = self.lm_head(x)  -> (B, T, vocab_size)
        10. If targets is not None:
              loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            Else:
              loss = None
        11. Return (logits, loss)
        """
        # TODO: Implement forward following the steps above
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
