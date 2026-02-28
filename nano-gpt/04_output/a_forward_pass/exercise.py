"""
GPT forward pass: turning token indices into logits and optional loss.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# PROVIDED: All sub-components (do not modify)
# ---------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
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
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_size, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
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
# PROVIDED: __init__ and _init_weights (do not modify)
# YOUR TASK: Implement forward()
# ---------------------------------------------------------------------------
class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
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
            logits: Tensor of shape (B, T, vocab_size) if targets is None,
                    or (B*T, vocab_size) if targets provided.
            loss: Scalar cross-entropy loss, or None if no targets.

        Steps:
        1. Extract B, T from idx.shape
        2. Compute tok_emb by looking up idx in the token embedding table -> (B, T, C)
        3. Compute pos_emb by creating a range of T positions on the same device as idx
           and looking up in the position embedding table (use torch.arange) -> (T, C)
        4. Combine: x = tok_emb + pos_emb -> (B, T, C)
        5. Pass x through the transformer blocks -> (B, T, C)
        6. Apply the final layer norm to x -> (B, T, C)
        7. Compute logits by passing x through the language model head -> (B, T, vocab_size)
        8. If targets is not None:
              Reshape logits to (B*T, vocab_size) and targets to (B*T),
              then compute loss using F.cross_entropy
           Else:
              loss = None
        9. Return (logits, loss)
        """
        # TODO: Implement the forward pass
        raise NotImplementedError("Implement forward")
