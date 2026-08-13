"""A tiny model used as an entry-point target by the meta-provider tests.

Kept deliberately small and dependency-light: its parameter count is checkable by hand,
so the meta provider's "exact" claim is verified against arithmetic rather than against
another copy of the same code.
"""

import torch
import torch.nn as nn


class Tiny(nn.Module):
    """Embedding + one linear layer. Parameters: vocab*hidden + hidden*hidden + hidden."""

    def __init__(self, vocab=100, hidden=32):
        super().__init__()
        self.vocab = vocab
        self.hidden = hidden
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.register_buffer("scale", torch.ones(hidden))

    def forward(self, input_ids):
        return self.proj(self.embed(input_ids)) * self.scale

    @staticmethod
    def expected_params(vocab=100, hidden=32):
        return vocab * hidden + hidden * hidden + hidden


def build_tiny(vocab=100, hidden=32):
    """Factory form of the same model, for testing --model-args."""
    return Tiny(vocab=vocab, hidden=hidden)


def build_frozen(vocab=100, hidden=32):
    """Same model with the embedding frozen, to test trainable-parameter counting."""
    model = Tiny(vocab=vocab, hidden=hidden)
    model.embed.weight.requires_grad_(False)
    return model


def not_a_model():
    return {"definitely": "not a module"}


NOT_CALLABLE = 42


class MiniTransformer(nn.Module):
    """A standard transformer stack, shaped so the analytic formula applies exactly.

    Used to cross-check the meta provider against ``params_from_transformer_shape``:
    two independent methods, one measuring and one deriving, must agree.
    """

    def __init__(self, layers=3, hidden=64, heads=4, intermediate=256, vocab=200):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList(
            [_Block(hidden, heads, intermediate) for _ in range(layers)]
        )
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class _Block(nn.Module):
    def __init__(self, hidden, heads, intermediate):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, intermediate, bias=False)
        self.fc2 = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        b, s, h = x.shape
        d = h // self.heads
        y = self.ln1(x)
        q = y @ self.q.weight.T
        k = y @ self.k.weight.T
        v = y @ self.v.weight.T
        q = q.view(b, s, self.heads, d).transpose(1, 2)
        k = k.view(b, s, self.heads, d).transpose(1, 2)
        v = v.view(b, s, self.heads, d).transpose(1, 2)
        att = ((q @ k.transpose(-2, -1)) / d ** 0.5).softmax(-1) @ v
        x = x + self.o(att.transpose(1, 2).reshape(b, s, h))
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(x))))
