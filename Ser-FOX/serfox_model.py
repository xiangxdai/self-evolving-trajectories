"""
Ser-FOX model definition built on a GPT-style decoder.

This module keeps standard autoregressive optimization, while exposing a
parallel index-scoring interface through shared frontier positions and an
index-isolating attention mask during serialization.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F


import os


def build_ste_visible_mask(seq_len, num_indices, device):
    """
    Build the Ser-FOX visibility mask for a sequence of the form:

        [serialized_prefix][appended_index_block]

    Visibility rules:
    - tokens inside the serialized prefix use standard causal visibility
    - each appended index token can attend to the full prefix
    - appended index tokens are isolated from one another and can only
      attend to themselves inside the appended block

    This implements the shared-frontier / index-isolation mask used by Ser-FOX.
    """
    if num_indices <= 0:
        raise ValueError(f"num_indices must be positive, got {num_indices}")
    if num_indices > seq_len:
        raise ValueError(f"num_indices ({num_indices}) cannot exceed seq_len ({seq_len})")

    prefix_len = seq_len - num_indices
    visible = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

    if prefix_len > 0:
        visible[:prefix_len, :prefix_len] = torch.tril(
            torch.ones((prefix_len, prefix_len), dtype=torch.bool, device=device)
        )
        visible[prefix_len:, :prefix_len] = True

    visible[prefix_len:, prefix_len:] = torch.eye(num_indices, dtype=torch.bool, device=device)
    return visible


def build_ste_group_visible_mask(seq_len, num_groups, group_size, device):
    """
    Build a Ser-FOX visibility mask for grouped parallel candidates:

        [serialized_prefix][group_1][group_2]...

    Each group can see the full prefix and previous tokens inside the same
    group, but cannot see tokens from other groups. This keeps candidate groups
    parallel while allowing multi-token indices such as V5's [d1, d2].
    """
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    suffix_len = num_groups * group_size
    if suffix_len > seq_len:
        raise ValueError(f"group suffix length ({suffix_len}) cannot exceed seq_len ({seq_len})")

    prefix_len = seq_len - suffix_len
    visible = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

    if prefix_len > 0:
        visible[:prefix_len, :prefix_len] = torch.tril(
            torch.ones((prefix_len, prefix_len), dtype=torch.bool, device=device)
        )
        visible[prefix_len:, :prefix_len] = True

    group_visible = torch.tril(torch.ones((group_size, group_size), dtype=torch.bool, device=device))
    for group_idx in range(num_groups):
        start = prefix_len + group_idx * group_size
        end = start + group_size
        visible[start:end, start:end] = group_visible
    return visible


def sample_positions_from_scores(position_scores, decode_mask, temperature=1.0, top_k=None):
    """
    Select one trajectory/index position from per-position confidence scores.

    The temperature here is intentionally applied only across unresolved
    trajectory positions. Value-token logits are scored separately and should
    not be divided by this temperature.
    """
    if position_scores.shape != decode_mask.shape:
        raise ValueError(
            f"position_scores shape {tuple(position_scores.shape)} must match "
            f"decode_mask shape {tuple(decode_mask.shape)}"
        )
    decode_mask = decode_mask.bool()
    if not torch.all(decode_mask.any(dim=1)):
        raise ValueError("Each batch row must have at least one unresolved position")

    if temperature is None or temperature <= 0:
        masked_scores = position_scores.masked_fill(~decode_mask, -float("Inf"))
        return masked_scores.argmax(dim=1)

    tiny = torch.finfo(position_scores.dtype).tiny
    logits = torch.log(position_scores.clamp_min(tiny)) / temperature
    logits = logits.masked_fill(~decode_mask, -float("Inf"))
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        logits = logits.masked_fill(logits < v[..., [-1]], -float("Inf"))
    probs = F.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # self.flash = False
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        self._ste_mask_cache = {}

    def _get_ste_visible_mask(self, seq_len, num_indices, device):
        key = (seq_len, num_indices, device.type, device.index)
        visible = self._ste_mask_cache.get(key)
        if visible is None:
            visible = build_ste_visible_mask(seq_len, num_indices, device)
            self._ste_mask_cache[key] = visible
        return visible

    def _project_qkv(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        return q, k, v

    def _finish_attention(self, y, B, T, C):
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

    def forward_causal_with_kv(self, x):
        """Manual causal attention that also returns the K/V tensors for caching."""
        B, T, C = x.size()
        q, k, v = self._project_qkv(x)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        return self._finish_attention(y, B, T, C), k, v

    def forward_new_tokens_with_kv(self, x, prefix_k, prefix_v, causal_new_tokens):
        """
        Run attention for newly appended tokens against cached prefix K/V.

        If causal_new_tokens is True, new tokens can also attend to previous new
        tokens in their local block. If False, new tokens are isolated from one
        another and can only attend to the prefix plus themselves, matching the
        Ser-FOX parallel index block.
        """
        B, T, C = x.size()
        q, k, v = self._project_qkv(x)
        scale = 1.0 / math.sqrt(k.size(-1))
        pieces = []
        values = []

        if prefix_k is not None and prefix_k.size(2) > 0:
            pieces.append((q @ prefix_k.transpose(-2, -1)) * scale)
            values.append(prefix_v)

        new_att = (q @ k.transpose(-2, -1)) * scale
        if causal_new_tokens:
            visible = torch.tril(torch.ones((T, T), dtype=torch.bool, device=x.device))
        else:
            visible = torch.eye(T, dtype=torch.bool, device=x.device)
        new_att = new_att.masked_fill(~visible.view(1, 1, T, T), float('-inf'))
        pieces.append(new_att)
        values.append(v)

        att = torch.cat(pieces, dim=-1)
        all_v = torch.cat(values, dim=2)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ all_v
        return self._finish_attention(y, B, T, C), k, v

    def forward(self, x, num_parallel_indices=None, visible_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self._project_qkv(x)

        # Fast path for standard causal AR attention.
        if self.flash and num_parallel_indices is None and visible_mask is None:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual path, kept for the custom Ser-FOX visibility mask.
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if visible_mask is not None:
                att = att.masked_fill(~visible_mask.view(1, 1, T, T), float('-inf'))
            elif num_parallel_indices is None:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            else:
                visible = self._get_ste_visible_mask(T, num_parallel_indices, att.device)
                att = att.masked_fill(~visible.view(1, 1, T, T), float('-inf'))


            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        return self._finish_attention(y, B, T, C)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, num_parallel_indices=None, visible_mask=None):
        x = x + self.attn(self.ln_1(x), num_parallel_indices, visible_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_causal_with_kv(self, x):
        attn_out, k, v = self.attn.forward_causal_with_kv(self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, k, v

    def forward_new_tokens_with_kv(self, x, prefix_k, prefix_v, causal_new_tokens):
        attn_out, k, v = self.attn.forward_new_tokens_with_kv(
            self.ln_1(x),
            prefix_k,
            prefix_v,
            causal_new_tokens,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, k, v


# ----------------------------- RoPE support -----------------------------
# Optional Rotary Position Embedding (RoPE) variant of Ser-FOX. When
# config.use_rope is True, the GPT uses RoPEBlock instead of Block and drops
# the learned absolute position embedding (wpe): positions enter only through
# RoPE applied to q/k inside attention. Everything else (token embedding,
# Ser-FOX index/value layout, parallel-index frontier sharing, KV-cache decode)
# is preserved.
#
# Key design points:
# - RoPE rotates q and k by their position index before attention. The KV cache
#   stores already-rotated k, so appending new (rotated) k keeps q.k dependent
#   only on relative offset — consistent with the recompute path.
# - score_parallel_indices preserves Ser-FOX's shared-frontier semantics: every
#   appended index token is rotated by the SAME frontier position (prefix_len),
#   mirroring the wpe shared-frontier in the absolute-PE path.
def build_rope_cache(head_dim, max_pos, theta=10000.0):
    """Return (cos, sin) each of shape (max_pos, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)              # (max_pos, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)       # (max_pos, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: (B, nh, T, hd); cos,sin: (T, hd)
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class RoPECausalSelfAttention(CausalSelfAttention):
    """CausalSelfAttention with RoPE applied to q/k. Positions are passed in
    explicitly so the Ser-FOX shared-frontier schedule can be honored."""

    def __init__(self, config):
        super().__init__(config)
        head_dim = config.n_embd // config.n_head
        cos, sin = build_rope_cache(head_dim, config.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _rope(self, q, k, positions):
        cos = self.rope_cos[positions]            # (T, hd)
        sin = self.rope_sin[positions]
        return apply_rope(q, k, cos, sin)

    def forward(self, x, positions, num_parallel_indices=None, visible_mask=None):
        B, T, C = x.size()
        q, k, v = self._project_qkv(x)
        q, k = self._rope(q, k, positions)
        if self.flash and num_parallel_indices is None and visible_mask is None:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0, is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if visible_mask is not None:
                att = att.masked_fill(~visible_mask.view(1, 1, T, T), float('-inf'))
            elif num_parallel_indices is None:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            else:
                visible = self._get_ste_visible_mask(T, num_parallel_indices, att.device)
                att = att.masked_fill(~visible.view(1, 1, T, T), float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        return self._finish_attention(y, B, T, C)

    def forward_causal_with_kv(self, x, positions):
        B, T, C = x.size()
        q, k, v = self._project_qkv(x)
        q, k = self._rope(q, k, positions)        # store rotated k in cache
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        return self._finish_attention(y, B, T, C), k, v

    def forward_new_tokens_with_kv(self, x, positions, prefix_k, prefix_v, causal_new_tokens):
        B, T, C = x.size()
        q, k, v = self._project_qkv(x)
        q, k = self._rope(q, k, positions)        # new q/k rotated by their own positions
        scale = 1.0 / math.sqrt(k.size(-1))
        pieces, values = [], []
        if prefix_k is not None and prefix_k.size(2) > 0:
            pieces.append((q @ prefix_k.transpose(-2, -1)) * scale)  # prefix_k already rotated
            values.append(prefix_v)
        new_att = (q @ k.transpose(-2, -1)) * scale
        if causal_new_tokens:
            visible = torch.tril(torch.ones((T, T), dtype=torch.bool, device=x.device))
        else:
            visible = torch.eye(T, dtype=torch.bool, device=x.device)
        new_att = new_att.masked_fill(~visible.view(1, 1, T, T), float('-inf'))
        pieces.append(new_att)
        values.append(v)
        att = torch.cat(pieces, dim=-1)
        all_v = torch.cat(values, dim=2)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ all_v
        return self._finish_attention(y, B, T, C), k, v


class RoPEBlock(Block):
    """Block that swaps in RoPECausalSelfAttention and threads positions through
    every forward variant."""

    def __init__(self, config):
        super().__init__(config)
        self.attn = RoPECausalSelfAttention(config)

    def forward(self, x, positions, num_parallel_indices=None, visible_mask=None):
        x = x + self.attn(self.ln_1(x), positions, num_parallel_indices, visible_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_causal_with_kv(self, x, positions):
        a, k, v = self.attn.forward_causal_with_kv(self.ln_1(x), positions)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, k, v

    def forward_new_tokens_with_kv(self, x, positions, prefix_k, prefix_v, causal_new_tokens):
        a, k, v = self.attn.forward_new_tokens_with_kv(
            self.ln_1(x), positions, prefix_k, prefix_v, causal_new_tokens)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, k, v


class GPTConfig:
    def __init__(
        self,
        vocab_size,
        block_size,
        n_layer,
        n_head,
        n_embd,
        dropout=0.1,
        bias=True,
        quiz_size=0,
        response_size=0,
        mask_token_id=None,
        dollar_token_id=None,
        value_vocab_size=None,
        dual_head=False,
        use_rope=False,
    ):
        # Total vocabulary size used by embeddings / lm_head: |Vser| = |V| + |Vspecial|
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.dual_head = dual_head  # unused in serialized-AR; retained for checkpoint back-compat
        # When True, use Rotary Position Embedding (RoPE) instead of the learned
        # absolute position embedding (wpe). Default False = unchanged behavior.
        self.use_rope = use_rope

        # Task layout
        self.quiz_size = quiz_size
        self.response_size = response_size

        # Reserved task-specific ids (unused in serialized-AR; retained for checkpoint back-compat)
        self.mask_token_id = mask_token_id
        self.dollar_token_id = dollar_token_id

        # Derived Ser-FOX sequence lengths
        # base_seq_len:
        #   raw task sample stored in base/test data, laid out as [prompt][response]
        #
        # train_seq_len:
        #   serialized training target, laid out as [prompt][I_i, y_i]...
        #
        # max_parallel_seq_len:
        #   longest phase-1 scoring input under the current implementation,
        #   where the current serialized prefix is followed by a full appended
        #   index block for parallel scoring
        self.num_index_tokens = response_size
        self.base_seq_len = quiz_size + response_size
        self.train_seq_len = quiz_size + 2 * response_size
        self.max_parallel_seq_len = quiz_size + 3 * response_size

        # Clean vocabulary size |V|
        self.value_vocab_size = (
            vocab_size - response_size if value_vocab_size is None else value_vocab_size
        )
        self.index_token_start = self.value_vocab_size

        assert self.vocab_size == self.value_vocab_size + self.num_index_tokens, (
            f"Expected vocab_size == value_vocab_size + num_index_tokens, "
            f"got {self.vocab_size} vs {self.value_vocab_size} + {self.num_index_tokens}"
        )
        if self.response_size > 0:
            assert self.block_size >= self.max_parallel_seq_len, (
                f"block_size={self.block_size} is too small for current Ser-FOX layout; "
                f"need at least {self.max_parallel_seq_len}"
            )

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        #####self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        if config.use_rope:
            # RoPE variant: swap in RoPE blocks and drop wpe entirely (RoPE needs
            # no absolute PE; positions enter only through rotation). Keeping an
            # unused wpe parameter would make DDP fail ("did not receive grad").
            self.transformer.h = nn.ModuleList([RoPEBlock(config) for _ in range(config.n_layer)])
            del self.transformer.wpe

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        if config.use_rope:
            print("number of parameters: %.2fM (RoPE)" % (self.get_num_params()/1e6,))
        else:
            print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def unused_index_mask(self, idx):
        """Return True for response indices that have not appeared in the serialized prefix."""
        cfg = self.config
        used_positions = idx - cfg.index_token_start
        valid = (used_positions >= 0) & (used_positions < cfg.num_index_tokens)
        decode = torch.ones(
            (idx.size(0), cfg.num_index_tokens),
            dtype=torch.bool,
            device=idx.device,
        )
        rows = torch.arange(idx.size(0), device=idx.device).unsqueeze(1).expand_as(idx)
        decode[rows[valid], used_positions[valid]] = False
        return decode

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.use_rope:
            # RoPE has no wpe parameter to subtract.
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_position_ids(self, idx):
        _, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        return torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)

    def _run_transformer(self, x, num_parallel_indices=None, visible_mask=None, positions=None):
        x = self.transformer.drop(x)
        if self.config.use_rope:
            for block in self.transformer.h:
                x = block(x, positions, num_parallel_indices, visible_mask)
        else:
            for block in self.transformer.h:
                x = block(x, num_parallel_indices, visible_mask)
        return self.transformer.ln_f(x)

    def build_prefix_kv_cache(self, idx):
        """Build a layer-wise K/V cache for a serialized causal prefix."""
        if self.config.use_rope:
            t = idx.size(1)
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            positions = torch.arange(0, t, dtype=torch.long, device=idx.device)
            x = self.transformer.drop(self.transformer.wte(idx))
            layers = []
            for block in self.transformer.h:
                x, k, v = block.forward_causal_with_kv(x, positions)
                layers.append({"k": k, "v": v})
            return {"layers": layers, "seq_len": idx.size(1)}

        pos = self._build_position_ids(idx)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        layers = []
        for block in self.transformer.h:
            x, k, v = block.forward_causal_with_kv(x)
            layers.append({"k": k, "v": v})

        return {"layers": layers, "seq_len": idx.size(1)}

    def append_to_kv_cache(self, cache, idx_new, return_hidden=False):
        """Append serialized prefix tokens to an existing causal K/V cache."""
        if idx_new.size(1) == 0:
            if return_hidden:
                return None, cache
            return cache

        start = cache["seq_len"]
        end = start + idx_new.size(1)
        if end > self.config.block_size:
            raise ValueError(f"Cannot append to cache length {end}; block_size is {self.config.block_size}")

        if self.config.use_rope:
            positions = torch.arange(start, end, dtype=torch.long, device=idx_new.device)
            x = self.transformer.drop(self.transformer.wte(idx_new))
            for layer_cache, block in zip(cache["layers"], self.transformer.h):
                x, k, v = block.forward_new_tokens_with_kv(
                    x, positions, layer_cache["k"], layer_cache["v"], causal_new_tokens=True)
                layer_cache["k"] = torch.cat([layer_cache["k"], k], dim=2)
                layer_cache["v"] = torch.cat([layer_cache["v"], v], dim=2)
            cache["seq_len"] = end
            if return_hidden:
                return self.transformer.ln_f(x), cache
            return cache

        pos = torch.arange(start, end, dtype=torch.long, device=idx_new.device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx_new)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for layer_cache, block in zip(cache["layers"], self.transformer.h):
            x, k, v = block.forward_new_tokens_with_kv(
                x,
                layer_cache["k"],
                layer_cache["v"],
                causal_new_tokens=True,
            )
            layer_cache["k"] = torch.cat([layer_cache["k"], k], dim=2)
            layer_cache["v"] = torch.cat([layer_cache["v"], v], dim=2)

        cache["seq_len"] = end
        if return_hidden:
            return self.transformer.ln_f(x), cache
        return cache

    def prefill_kv_cache(self, idx):
        """Run a full causal prefix once and return its hidden states plus K/V cache."""
        if self.config.use_rope:
            t = idx.size(1)
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            positions = torch.arange(0, t, dtype=torch.long, device=idx.device)
            x = self.transformer.drop(self.transformer.wte(idx))
            layers = []
            for block in self.transformer.h:
                x, k, v = block.forward_causal_with_kv(x, positions)
                layers.append({"k": k, "v": v})
            return self.transformer.ln_f(x), {"layers": layers, "seq_len": idx.size(1)}

        pos = self._build_position_ids(idx)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        layers = []
        for block in self.transformer.h:
            x, k, v = block.forward_causal_with_kv(x)
            layers.append({"k": k, "v": v})

        return self.transformer.ln_f(x), {"layers": layers, "seq_len": idx.size(1)}

    def forward_ar(self, idx, targets=None):
        """Phase 2: standard causal AR fitting or AR inference."""
        if self.config.use_rope:
            _, t = idx.size()
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            positions = torch.arange(0, t, dtype=torch.long, device=idx.device)
            x = self._run_transformer(self.transformer.wte(idx), positions=positions)
        else:
            pos = self._build_position_ids(idx)
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos)
            x = self._run_transformer(tok_emb + pos_emb)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return logits, loss

    def score_parallel_indices(self, idx, num_indices):
        """Phase 1: score the appended index block under the Ser-FOX mask."""
        pos = self._build_position_ids(idx)
        _, t = idx.size()
        if num_indices <= 0:
            raise ValueError(f"num_indices must be positive, got {num_indices}")
        if num_indices > t:
            raise ValueError(f"num_indices ({num_indices}) cannot exceed sequence length ({t})")

        prefix_len = t - num_indices

        if self.config.use_rope:
            # Prefix uses sequential positions; every appended index token shares
            # the SAME frontier position (prefix_len) — identical role to the
            # wpe shared-frontier in the absolute-PE path below.
            positions = torch.cat([
                torch.arange(prefix_len, dtype=torch.long, device=idx.device),
                torch.full((num_indices,), prefix_len, dtype=torch.long, device=idx.device),
            ])
            x = self._run_transformer(
                self.transformer.wte(idx), num_parallel_indices=num_indices, positions=positions)
        else:
            # All appended index tokens share the same logical frontier position
            # while remaining token-distinct through their learned identities.
            tok_emb = self.transformer.wte(idx)
            prefix_pos_emb = self.transformer.wpe(pos[:, :prefix_len])
            frontier_pos_emb = self.transformer.wpe(pos[:, prefix_len:prefix_len+1]).repeat(1, num_indices, 1)
            pos_emb = torch.cat([prefix_pos_emb, frontier_pos_emb], dim=1)

            x = self._run_transformer(tok_emb + pos_emb, num_parallel_indices=num_indices)

        # Phase-1 keeps the full vocabulary shape for interface compatibility,
        # but special index tokens must never be emitted as candidate values.
        # We therefore hard-mask Vspecial to -inf here.
        logits = self.lm_head(x[:, -num_indices:, :]).clone()
        logits[..., self.config.index_token_start:] = float("-inf")
        return logits

    def score_parallel_digit_groups(self, idx, num_groups, group_size):
        """Score grouped parallel candidates such as V5 fixed-length digits.

        The appended suffix is interpreted as `num_groups` consecutive groups
        of `group_size` tokens. Groups share the same logical frontier position
        schedule and are isolated from one another; tokens inside a group use
        causal visibility so later digits can see earlier digits.
        """
        if self.config.use_rope:
            raise NotImplementedError(
                "RoPE Ser-FOX does not implement digit-group scoring (not used by sat9/sudoku).")
        if num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {num_groups}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")

        _, t = idx.size()
        suffix_len = num_groups * group_size
        if suffix_len > t:
            raise ValueError(f"group suffix length ({suffix_len}) cannot exceed sequence length ({t})")
        prefix_len = t - suffix_len

        pos = self._build_position_ids(idx)
        tok_emb = self.transformer.wte(idx)
        prefix_pos_emb = self.transformer.wpe(pos[:, :prefix_len])
        group_pos = torch.arange(
            prefix_len,
            prefix_len + group_size,
            dtype=torch.long,
            device=idx.device,
        ).repeat(num_groups).unsqueeze(0)
        group_pos_emb = self.transformer.wpe(group_pos)
        pos_emb = torch.cat([prefix_pos_emb, group_pos_emb], dim=1)

        visible = build_ste_group_visible_mask(t, num_groups, group_size, idx.device)
        x = self._run_transformer(tok_emb + pos_emb, visible_mask=visible)

        group_end_positions = prefix_len + torch.arange(
            group_size - 1,
            suffix_len,
            group_size,
            dtype=torch.long,
            device=idx.device,
        )
        logits = self.lm_head(x[:, group_end_positions, :]).clone()
        logits[..., self.config.index_token_start:] = float("-inf")
        return logits

    def score_parallel_indices_cached(self, index_tokens, cache):
        """Phase 1 scorer using a cached serialized prefix."""
        num_indices = index_tokens.size(1)
        if num_indices <= 0:
            raise ValueError(f"num_indices must be positive, got {num_indices}")
        if num_indices != self.config.num_index_tokens:
            raise ValueError(
                f"Expected {self.config.num_index_tokens} index tokens, got {num_indices}"
            )

        prefix_len = cache["seq_len"]
        if prefix_len + num_indices > self.config.block_size:
            raise ValueError(
                f"Cached scoring sequence length {prefix_len + num_indices} exceeds block_size {self.config.block_size}"
            )

        if self.config.use_rope:
            positions = torch.full((num_indices,), prefix_len, dtype=torch.long, device=index_tokens.device)
            x = self.transformer.drop(self.transformer.wte(index_tokens))
            for layer_cache, block in zip(cache["layers"], self.transformer.h):
                x, _, _ = block.forward_new_tokens_with_kv(
                    x, positions, layer_cache["k"], layer_cache["v"], causal_new_tokens=False)
        else:
            pos = torch.full((1, num_indices), prefix_len, dtype=torch.long, device=index_tokens.device)
            tok_emb = self.transformer.wte(index_tokens)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)

            for layer_cache, block in zip(cache["layers"], self.transformer.h):
                x, _, _ = block.forward_new_tokens_with_kv(
                    x,
                    layer_cache["k"],
                    layer_cache["v"],
                    causal_new_tokens=False,
                )

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x).clone()
        logits[..., self.config.index_token_start:] = float("-inf")
        return logits

    def forward(self, idx, targets=None):
        """
        Default PyTorch entry point.

        We intentionally keep forward() mapped to the standard autoregressive
        path so that model(idx, targets) behaves like a regular GPT call under
        DDP, torch.compile, checkpoint loading, and generic PyTorch tooling.

        Ser-FOX's parallel index scorer is exposed explicitly through
        score_parallel_indices().
        """
        return self.forward_ar(idx, targets)

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if not self.config.use_rope:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        
        
        #####decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate_parallel_index(self, idx, max_new_tokens, temperature=1.0, top_k=None, verbose=False, use_cache=True):
        """
        Mode 2: confidence-guided parallel-index decoding.

        At each step:
        1) append the full index block
        2) score all unresolved indices in parallel
        3) choose the best value for every index independently
        4) sample/rank the trajectory position using the per-position confidence
           scores; temperature/top-k apply only to this position selection
           and commit the best [index, value] pair back to the prefix

        KV-cache fast path (use_cache=True, default): the committed prefix grows
        by exactly one [index, value] pair (2 tokens) per step, so its K/V is
        built once and extended incrementally with append_to_kv_cache instead of
        re-encoding the whole prefix every step. This is mathematically identical
        to the recompute path because positions are sequential (arange) and the
        appended index block attends only to the prefix while being isolated from
        one another (causal_new_tokens=False in score_parallel_indices_cached).
        Falls back to the recompute path when the full decoded sequence would
        exceed block_size.
        """
        num_index_tokens = self.config.num_index_tokens
        device = idx.device
        a = idx.size(0)
        index_tokens = torch.arange(0, num_index_tokens, dtype=torch.long, device=device).unsqueeze(0)
        index_tokens = (index_tokens + self.config.index_token_start).expand(a, -1)

        can_cache = use_cache and (idx.size(1) + 2 * max_new_tokens <= self.config.block_size)
        kv_cache = self.build_prefix_kv_cache(idx) if can_cache else None

        for _ in range(max_new_tokens):
            if kv_cache is None:
                # Recompute path: re-encode the (cropped) prefix + index block.
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                idx_cond_app = torch.cat([idx_cond, index_tokens], dim=1)
                index_logits = self.score_parallel_indices(idx_cond_app, num_index_tokens)
                decode = self.unused_index_mask(idx_cond)
            else:
                # Cached path: score the index block against the cached prefix K/V.
                index_logits = self.score_parallel_indices_cached(index_tokens, kv_cache)
                decode = self.unused_index_mask(idx)

            permanent_probs = F.softmax(index_logits, dim=-1)
            idx_next = index_logits.argmax(dim=-1)

            if verbose:
                print(idx_next)

            p = permanent_probs.gather(dim=2, index=idx_next.unsqueeze(-1)).squeeze(-1)
            if verbose:
                print(p)

            max_idx = sample_positions_from_scores(
                p,
                decode,
                temperature=temperature,
                top_k=top_k,
            )
            next_1 = index_tokens.gather(1, max_idx.unsqueeze(1))
            next_2 = idx_next.gather(1, max_idx.unsqueeze(1))

            # Append the selected [index, value] pair back to the serialized prefix.
            idx = torch.cat([idx, next_1, next_2], dim=1)
            if kv_cache is not None:
                kv_cache = self.append_to_kv_cache(kv_cache, torch.cat([next_1, next_2], dim=1))

            if verbose:
                print(idx[0, num_index_tokens:])
                print("")
        return idx

    @torch.no_grad()
    def generate_serialized_ar(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        verbose=False,
        use_cache=True,
    ):
        """Mode 1: serialized autoregressive decoding."""
        if not use_cache or idx.size(1) + max_new_tokens > self.config.block_size:
            for _ in range(max_new_tokens):
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _ = self.forward_ar(idx_cond)
                logits = logits[:, -1, :] / temperature

                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)

                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                # Append the sampled next token to the serialized AR sequence.
                idx = torch.cat((idx, idx_next), dim=1)
                if verbose:
                    print(idx)
            return idx

        hidden, kv_cache = self.prefill_kv_cache(idx)
        next_logits = self.lm_head(hidden[:, -1, :])
        for _ in range(max_new_tokens):
            logits = next_logits / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append the sampled next token to the serialized AR sequence.
            idx = torch.cat((idx, idx_next), dim=1)
            if verbose:
                print(idx)
            hidden, kv_cache = self.append_to_kv_cache(kv_cache, idx_next, return_hidden=True)
            next_logits = self.lm_head(hidden[:, -1, :])
        return idx

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, verbose=False):
        """
        Legacy alias for Mode 2 parallel-index decoding.

        New code should call generate_parallel_index() explicitly.
        """
        return self.generate_parallel_index(idx, max_new_tokens, temperature=temperature, top_k=top_k, verbose=verbose)

    @torch.no_grad()
    def generate_all(self, idx, max_new_tokens, temperature=1.0, top_k=None, verbose=False):
        """
        Legacy alias for Mode 1 serialized autoregressive decoding.

        New code should call generate_serialized_ar() explicitly.
        """
        return self.generate_serialized_ar(idx, max_new_tokens, temperature=temperature, top_k=top_k, verbose=verbose)
