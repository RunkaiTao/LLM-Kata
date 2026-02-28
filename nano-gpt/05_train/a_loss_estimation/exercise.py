"""
Loss estimation for monitoring training progress.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# PROVIDED: All model components (do not modify)
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


# ---------------------------------------------------------------------------
# PROVIDED: get_batch (do not modify)
# ---------------------------------------------------------------------------
def get_batch(data, block_size, batch_size, device="cpu"):
    """Sample a random batch of input-target pairs."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# ---------------------------------------------------------------------------
# YOUR TASK: Implement estimate_loss
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device="cpu"):
    """
    Estimate the average loss on train and val splits.

    Args:
        model: The GPT model.
        train_data: Training data tensor.
        val_data: Validation data tensor.
        block_size: Context window size.
        batch_size: Batch size for evaluation.
        eval_iters: Number of batches to average over.
        device: Device to use ('cpu' or 'cuda').

    Returns:
        A dict {'train': mean_train_loss, 'val': mean_val_loss}

    Steps:
    1. Initialize output dict.
    2. Set model to eval mode: model.eval()
    3. For each split in ['train', 'val']:
       a. Select data: data = train_data if split == 'train' else val_data.
       b. Create a tensor of zeros with shape (eval_iters,) to accumulate losses.
       c. For k in range(eval_iters):
          - Get a batch: X, Y = get_batch(data, block_size, batch_size, device)
          - Run model forward: logits, loss = model(X, Y)
          - Store loss.item() in losses[k].
       d. Store the mean loss in the output dict.
    4. Set model back to train mode: model.train()
    5. Return the output dict.
    """
    # TODO: Implement loss estimation
    raise NotImplementedError("Implement estimate_loss")
