"""
NemotronH for Causal Language Modeling — the full model with LM head.

This is the top-level model that wraps NemotronHModel with a language model
head for next-token prediction. It adds:

1. An LM head (linear projection from hidden_size -> vocab_size)
2. Optional weight tying between token embeddings and LM head
3. Loss computation (cross-entropy) when targets are provided

The forward pass:
    input_ids -> NemotronHModel -> hidden_states -> lm_head -> logits
    If targets provided: compute cross-entropy loss

Reference: vllm/vllm/model_executor/models/nemotron_h.py lines 784-957
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

NemotronHConfig = load("01_config_and_primitives", "a_nemotron_h_config").NemotronHConfig
NemotronHModel = load("03_decoder_and_model", "b_nemotron_h_model").NemotronHModel


# ---------------------------------------------------------------------------
# YOUR TASK: Implement NemotronHForCausalLM
# ---------------------------------------------------------------------------

class NemotronHForCausalLM(nn.Module):
    """NemotronH with a language model head for causal (autoregressive) generation."""

    def __init__(self, config: "NemotronHConfig"):
        """
        Args:
            config: NemotronHConfig instance.

        Steps:
        1. Store config as self.config
        2. Create self.model as a NemotronHModel(config) instance
        3. Create self.lm_head as nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        4. If config.tie_word_embeddings is True:
           Tie the weights: self.model.embed_tokens.weight = self.lm_head.weight
           (so the embedding and output projection share the same parameter)
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        # Step 1: self.config = ...
        # Step 2: self.model = ...    (NemotronHModel instance)
        # Step 3: self.lm_head = ...  (nn.Linear: hidden_size -> vocab_size, no bias)
        # Step 4: if config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_ids: Token indices of shape (B, T).
            targets: Target token indices of shape (B, T), or None.

        Returns:
            logits: Tensor of shape (B, T, vocab_size).
            loss: Scalar cross-entropy loss, or None if no targets.

        Steps:
        1. Pass input_ids through self.model to get hidden_states -> (B, T, hidden_size)
        2. Project through self.lm_head to get logits -> (B, T, vocab_size)
        3. If targets is not None, compute cross-entropy loss after flattening
           logits and targets (use F.cross_entropy); otherwise loss is None
        4. Return (logits, loss)
        """
        # TODO: Implement forward following the steps above
        # Step 1: hidden_states = ...  (pass input_ids through self.model)
        # Step 2: logits = ...         (project through lm_head)
        # Step 3: loss = None          (if targets: compute F.cross_entropy on flattened logits/targets)
        # return logits, loss
        pass


# Run tests:
# pytest nano-nemotron/03_decoder_and_model/c_causal_lm/test_exercise.py -v
