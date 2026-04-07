"""
Training entrypoint for the 3Seg-PrefixVisible ablation.

This ablation keeps the same three-segment packed sequence as Tom-CAT, but uses
standard causal visibility inside the prediction segment rather than Tom-CAT's
teacherless write-space masking.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import argparse
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from three_seg_prefix_visible_model import GPTConfig, GPT
from logger import get_logger
import logging

# Command-line arguments.
parser = argparse.ArgumentParser(description='Train the Tom-CAT 3Seg-PrefixVisible ablation')
parser.add_argument('--dataset', type=str, default='cd5$', help='Name of the dataset to use')
parser.add_argument('--n_layer', type=int, default=12, help='Number of layers')
parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads')
parser.add_argument('--n_embd', type=int, default=768, help='Size of the embeddings')
parser.add_argument('--max_iters', type=int, default=1000000, help='Number of iterations')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
parser.add_argument('--compile', type=bool, default=True, help='Use PyTorch 2.0 compilation')
parser.add_argument('--suffix', type=str, default='0-1', help='Suffix to add to output directory')
args = parser.parse_args()


# Extract arguments.
dataset = args.dataset
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
max_iters = args.max_iters
train_batch_size = args.batch_size
val_batch_size = args.batch_size // 2
suffix = args.suffix
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
data_dir = os.path.join(repo_root, 'data', dataset)

# Load dataset metadata.
with open(os.path.join(data_dir, 'cd_meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
data_size = meta['data_size']
quiz_size = meta['quiz_size']
response_size = meta['response_size']
block_size = quiz_size + response_size * 2 - 1
pad_token_id = stoi['<PAD>'] 
eos_token_id = stoi['<EOS>']  
sep_token_id = stoi['<SEP>'] 
mask_token_id = stoi['<MASK>'] 
dollar_token_id = stoi['$']
out_dir = os.path.join(
    repo_root,
    'out',
    'tomcat_ablations',
    'three_seg_prefix_visible',
    f'{dataset}_{n_layer}_{n_head}_{n_embd}_{suffix}_{max_iters}',
)

# Default training configuration.
eval_interval = max_iters // 10
log_interval = max_iters // 100
eval_iters = max_iters // 10
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
wandb_log = False
wandb_project = 'cd'
wandb_run_name = f'tomcat-three-seg-prefix-visible-{dataset}-{suffix}'
gradient_accumulation_steps = 1
dropout = 0.1
bias = False
learning_rate = 3e-4

# -----------------------------------------------------------------------------
# Linear learning-rate scaling rule.
base_batch_size = 512
base_learning_rate = 3e-4

# Scale the learning rate with batch size to stay close to the reference setup.
learning_rate = base_learning_rate * (train_batch_size / base_batch_size)
print(f"Using scaled learning rate = {learning_rate:.6f} for batch size = {train_batch_size}")

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = max_iters // 20
lr_decay_iters = max_iters
min_lr = learning_rate / 10
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16'
compile = args.compile

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

# DDP setup.
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % torch.cuda.device_count() == 0
    gradient_accumulation_steps //= torch.cuda.device_count()
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * train_batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loader.
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    batch_size = train_batch_size if split == 'train' else val_batch_size
    ignore_index = -100
    vocab_size = meta['vocab_size']

    # Packed layout:
    #   x = [quiz] [masked_response] [response[:-1]]
    #   y = [ignore on quiz and masked second segment] [response]
    #
    # This ablation matches Tom-CAT's three-segment data layout, but keeps the
    # standard causal mask in the model rather than blocking write-space
    # prefix visibility.

    # Make sure sampled windows stay inside one serialized example.
    max_index = len(data) - data_size
    if max_index < 0:
        raise ValueError(f"Data size {len(data)} is smaller than data_size {data_size}")

    # Sample example-aligned starting indices.
    ix = torch.randint(0, (max_index // data_size) + 1, (batch_size,)) * data_size

    # Split the serialized example into the quiz and reference response.
    quizzes = torch.stack([torch.from_numpy(data[i:i+quiz_size].astype(np.int64)) for i in ix])
    responses = torch.stack([torch.from_numpy(data[i+quiz_size:i+data_size].astype(np.int64)) for i in ix])

    # Each sample draws its own mask ratio for the second segment.
    mask_ratios = torch.empty(batch_size).uniform_(0.001, 1)

    # # Optional: use a discrete set of mask ratios instead.
    # possible_ratios = torch.arange(0.1, 1.0 + 1e-8, 0.2)  # [0.1, 0.2, ..., 1.0]

    # # Randomly pick one ratio per sample from the discrete set.
    # mask_ratios = possible_ratios[torch.randint(len(possible_ratios), (batch_size,))]


    # Initialize the second-segment mask.
    mask = torch.zeros((batch_size, response_size), dtype=torch.bool)

    for b in range(batch_size):
        # Decide how many positions to mask in this sample.
        num_to_mask = max(1, int(mask_ratios[b].item() * response_size))

        # Pick the masked positions uniformly at random.
        mask_positions = torch.randperm(response_size)[:num_to_mask]
        mask[b, mask_positions] = True

    # # Optional: use a fixed full mask instead.
    # mask_probs = torch.ones(batch_size, 1)
    # mask = torch.rand(batch_size, response_size) < mask_probs

    # # Optional: sample an independent Bernoulli mask probability per example.
    # # Restrict each sample to a probability range such as [0.2, 0.6].
    # mask_probs = torch.empty(batch_size, 1).uniform_(0.2, 0.6)
    # # Then sample the actual binary mask.
    # mask = torch.rand(batch_size, response_size) < mask_probs

    # Build the masked second segment.
    response_masked = responses.clone()
    # Replace the terminal <EOS> with <SEP> because this is the conditioning segment.
    response_masked[:, -1] = sep_token_id
    # Replace sampled positions with the shared mask token.
    response_masked[mask] = mask_token_id

    # Construct x = quiz + masked_response + response[:-1].
    x = torch.zeros((batch_size, block_size), dtype=torch.int64)
    x[:, :quiz_size] = quizzes
    x[:, quiz_size:quiz_size+response_size] = response_masked
    x[:, quiz_size+response_size:block_size] = responses[:, :-1]

    # Only supervise the final write segment in this ablation.
    y = torch.full((batch_size, block_size), ignore_index, dtype=torch.int64)

    # The earlier segments are ignored; the last segment predicts the full response.
    y[:, quiz_size-1+response_size:block_size] = responses

    # Move the batch to the active device.
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)


    # Optional decode helper for debugging packed batches.
    def decode(tokens):
        return ''.join([itos[int(t)] for t in tokens.cpu().tolist() if int(t) in itos])

    # print("x shape:", x.shape)
    # print("y shape:", y.shape)
    # print("x[0] decoded:", decode(x[0]))
    # print("y[0] decoded:", decode(y[0]))


    return x, y


# Logger.
logger = get_logger(os.path.join(out_dir, 'train.log'))
log_file_name = os.path.join(out_dir, 'train.log')

# Load vocab size.
meta_path = os.path.join(data_dir, 'cd_meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Model initialization.
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=meta_vocab_size, dropout=dropout,
                  quiz_size=quiz_size, response_size=response_size, 
                  mask_token_id=mask_token_id, dollar_token_id=dollar_token_id)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
model.to(device)

# GradScaler for mixed precision.
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer.
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# Compile model.
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# DDP wrapper.
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Estimate loss.
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Learning-rate scheduler.
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# File append utility.
def open_and_append(filename, text):
    with open(filename, 'a') as file:
        file.write(text + '\n')

# WandB logging.
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Training loop.
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
iter_num = 0
best_val_loss = 1e9

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        open_and_append(log_file_name, f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                logger.info(f"saving checkpoint to {out_dir}")
                open_and_append(log_file_name, f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, f'{iter_num}_ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(train_batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        logger.info(f"iter {iter_num}: loss {lossf:.4f}")
        open_and_append(log_file_name, f"iter {iter_num}: loss {lossf:.4f}")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
