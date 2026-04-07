"""
Standard causal language-model baseline used for autoregressive comparisons.

This module keeps a plain decoder-only Transformer with standard left-to-right
attention and token-by-token generation. It does not introduce Tom-CAT's masked
second segment, teacherless write space, or any Ser-FOX serialization machinery.
"""

import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def new_gelu(x):
    """Approximate GELU used by GPT/BERT-style models."""
    return 0.5 * x * (
        1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
    )


class LayerNorm(nn.Module):
    """LayerNorm with optional bias, matching the NanoGPT style."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input_tensor):
        return F.layer_norm(input_tensor, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        batch_size, seq_len, channels = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(batch_size, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
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

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
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
        quiz_size=15,
        response_size=35,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.quiz_size = quiz_size
        self.response_size = response_size


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for parameter_name, parameter in self.named_parameters():
            if parameter_name.endswith("c_proj.weight"):
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        self.vocab_size = config.vocab_size
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """Return the parameter count, optionally excluding position embeddings."""
        num_params = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            num_params -= self.transformer.wpe.weight.numel()
        return num_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        batch_size, seq_len = idx.size()
        assert (
            seq_len <= self.config.block_size
        ), f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"

        token_embeddings = self.transformer.wte(idx)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        position_embeddings = self.transformer.wpe(positions)
        x = self.transformer.drop(token_embeddings + position_embeddings)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """Shrink the maximum supported context length if needed."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(key == "dropout" for key in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]

        config = GPTConfig(**config_args)
        model = GPT(config)
        state_dict = model.state_dict()
        state_dict_keys = [key for key in state_dict.keys() if not key.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        state_dict_hf = model_hf.state_dict()

        state_dict_hf_keys = [key for key in state_dict_hf.keys() if not key.endswith(".attn.masked_bias")]
        state_dict_hf_keys = [key for key in state_dict_hf_keys if not key.endswith(".attn.bias")]
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        assert len(state_dict_hf_keys) == len(state_dict_keys), (
            f"mismatched keys: {len(state_dict_hf_keys)} != {len(state_dict_keys)}"
        )
        for key in state_dict_hf_keys:
            if any(key.endswith(weight_name) for weight_name in transposed):
                assert state_dict_hf[key].shape[::-1] == state_dict[key].shape
                with torch.no_grad():
                    state_dict[key].copy_(state_dict_hf[key].t())
            else:
                assert state_dict_hf[key].shape == state_dict[key].shape
                with torch.no_grad():
                    state_dict[key].copy_(state_dict_hf[key])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Create AdamW with decayed and non-decayed parameter groups."""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for module_name, module in self.named_modules():
            for parameter_name, _ in module.named_parameters():
                full_parameter_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
                if parameter_name.endswith("bias"):
                    no_decay.add(full_parameter_name)
                elif parameter_name.endswith("weight") and isinstance(module, whitelist_weight_modules):
                    decay.add(full_parameter_name)
                elif parameter_name.endswith("weight") and isinstance(module, blacklist_weight_modules):
                    no_decay.add(full_parameter_name)

        decay.remove("lm_head.weight")

        parameter_dict = {parameter_name: parameter for parameter_name, parameter in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        )
        assert len(parameter_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(parameter_dict.keys() - union_params),)
        )

        optimizer_groups = [
            {
                "params": [parameter_dict[name] for name in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [parameter_dict[name] for name in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(optimizer_groups, lr=learning_rate, betas=betas, **extra_args)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 peak BF16 FLOPS."""
        num_params = self.get_num_params()
        cfg = self.config
        num_layers = cfg.n_layer
        num_heads = cfg.n_head
        head_dim = cfg.n_embd // cfg.n_head
        context_length = cfg.block_size
        flops_per_token = 6 * num_params + 12 * num_layers * num_heads * head_dim * context_length
        flops_per_fwdbwd = flops_per_token * context_length
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Standard left-to-right autoregressive generation."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
