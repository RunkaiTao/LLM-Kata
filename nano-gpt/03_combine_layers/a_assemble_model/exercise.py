"""
Assembling the full GPT language model from sub-components.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# PROVIDED: All sub-components (do not modify)
# ---------------------------------------------------------------------------
class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, n_embd: int, n_head: int, head_size: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# YOUR TASK: Implement GPTLanguageModel.__init__ and _init_weights
# ---------------------------------------------------------------------------
class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size: int, block_size: int, n_embd: int, n_head: int, n_layer: int, dropout: float = 0.0):
        """
        Args:
            vocab_size: Number of tokens in the vocabulary.
            block_size: Maximum sequence length.
            n_embd: Embedding dimension.
            n_head: Number of attention heads.
            n_layer: Number of transformer blocks.
            dropout: Dropout rate.
        """
        super().__init__()
        # TODO: Store block_size as self.block_size for later use
        # TODO: Create self.token_embedding_table as an nn.Embedding mapping vocab_size to n_embd
        # TODO: Create self.position_embedding_table as an nn.Embedding mapping block_size to n_embd
        # TODO: Create self.blocks as an nn.Sequential of n_layer Block instances
        #       (each with n_embd, n_head, block_size, dropout)
        # TODO: Create self.ln_f as an nn.LayerNorm over n_embd dimensions
        # TODO: Create self.lm_head as a linear layer from n_embd to vocab_size (use nn.Linear)
        # TODO: Apply weight initialization by calling self.apply with self._init_weights
        raise NotImplementedError("Implement __init__")

    def _init_weights(self, module):
        """
        Initialize weights for the model.

        Rules:
        - If module is nn.Linear: set weights to Normal(mean=0.0, std=0.02),
          and if bias exists, set to zeros.
        - If module is nn.Embedding: set weights to Normal(mean=0.0, std=0.02).

        Hint: Use torch.nn.init.normal_() and torch.nn.init.zeros_()
        """
        # TODO: Implement weight initialization
        raise NotImplementedError("Implement _init_weights")

    def forward(self, idx, targets=None):
        """PROVIDED: Forward pass (for testing the model assembly)."""
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
