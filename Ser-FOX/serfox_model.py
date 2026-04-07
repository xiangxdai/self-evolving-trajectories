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


image_counter = {"count": 0}

def save_attention_images_ordered(att, output_dir="att_images", counter=image_counter):
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(att)):
        plt.imshow(att[i].cpu().numpy(), cmap='viridis')
        plt.colorbar()
        filename = f"{counter['count']}_head{i}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
    counter['count'] += 1

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

    def forward(self, x, num_parallel_indices=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Fast path for standard causal AR attention.
        if self.flash and num_parallel_indices is None:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual path, kept for the custom Ser-FOX visibility mask.
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if num_parallel_indices is None:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            else:
                visible = build_ste_visible_mask(T, num_parallel_indices, att.device)
                att = att.masked_fill(~visible.view(1, 1, T, T), float('-inf'))


            att = F.softmax(att, dim=-1)
            #print(att[0,0,24,:])
            #print(att[0,0,17,:])
            #save_attention_images_ordered(att[0,:,:,:])
            
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

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

    def forward(self, x, num_parallel_indices=None):
        x = x + self.attn(self.ln_1(x), num_parallel_indices)
        x = x + self.mlp(self.ln_2(x))
        return x

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
    ):
        # Total vocabulary size used by embeddings / lm_head: |Vser| = |V| + |Vspecial|
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias

        # Task layout
        self.quiz_size = quiz_size
        self.response_size = response_size

        # Reserved task-specific ids
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

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
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

    def _run_transformer(self, x, num_parallel_indices=None):
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, num_parallel_indices)
        return self.transformer.ln_f(x)

    def forward_ar(self, idx, targets=None):
        """Phase 2: standard causal AR fitting or AR inference."""
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
    def generate_parallel_index(self, idx, max_new_tokens, temperature=1.0, top_k=None, verbose=False):
        """
        Mode 2: confidence-guided parallel-index decoding.

        At each step:
        1) append the full index block
        2) score all unresolved indices in parallel
        3) sample candidate values for every index from the temperature/top-k
           proposal distribution
        4) rank those sampled candidates using the model's original confidence
           and commit the best [index, value] pair back to the prefix
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            a, b = idx_cond.shape
            num_index_tokens = self.config.num_index_tokens
            device = idx.device

            index_tokens = torch.arange(0, num_index_tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(a, 1)
            index_tokens = index_tokens + self.config.index_token_start
            idx_cond_app = torch.cat([idx_cond, index_tokens], dim=1)

            index_logits = self.score_parallel_indices(idx_cond_app, num_index_tokens)
            permanent_probs = F.softmax(index_logits, dim=-1)
            logits = index_logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits = logits.masked_fill(logits < v[..., [-1]], -float('Inf'))
            probs = F.softmax(logits, dim=-1)

            eq = idx_cond_app[:, :, None] == idx_cond_app[:, None, :]      # (a, b, b) 逐行两两比较
            has_previous_same = eq.tril(diagonal=-1).any(dim=-1)
            decode = ~has_previous_same                # True for the first occurrence only
            decode = decode[:, -num_index_tokens:]

            probs = probs.view(a * num_index_tokens, self.config.vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = idx_next.view(a, num_index_tokens)

            if verbose:
                print(idx_next)

            p = permanent_probs.gather(dim=2, index=idx_next.unsqueeze(-1)).squeeze(-1)
            p = torch.where(decode, p, -1)
            if verbose:
                print(p)

            _, max_idx = p.max(dim=1)   # shape: (a, 1)
            next_1 = index_tokens.gather(1, max_idx.unsqueeze(1))
            next_2 = idx_next.gather(1, max_idx.unsqueeze(1))

            # Append the selected [index, value] pair back to the serialized prefix.
            idx = torch.cat([idx, next_1, next_2], dim=1)

            if verbose:
                print(idx[0, num_index_tokens:])
                print("")
        return idx

    @torch.no_grad()
    def generate_serialized_ar(self, idx, max_new_tokens, temperature=1.0, top_k=None, verbose=False):
        """Mode 1: serialized autoregressive decoding."""
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
