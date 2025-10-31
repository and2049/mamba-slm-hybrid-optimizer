import torch

wandb_project_name = 'mamba-slm-finetuning'

pretrained_checkpoint_path = "checkpoints/best.pt"

# Model config (MUST match pre-training)
d_model = 768
n_layer = 24
vocab_size = 50277
pad_vocab_size_multiple = 8

max_seq_len = 1024

# fine-tuning config
limit_train_batches = None
limit_val_batches = None

learning_rate_muon = 5e-5   # e.g., 5e-5
learning_rate_adamw = 1e-5  # e.g., 1e-5
weight_decay = 0.01

gradient_accumulation_steps = 32
micro_batch_size = 1

# calculate tokens per step
# (max_seq_len * micro_batch_size * grad_accum_steps)
tokens_per_step = max_seq_len * micro_batch_size * gradient_accumulation_steps

# --- Training Duration (Epoch based) ---
# total tokens in the Alpaca dataset (approx. 52k samples * ~200 tokens/sample)
total_train_tokens = 10_400_000

# train by epochs
train_by_epochs = True
num_epochs = 2            # 1-3 epochs is typical for SFT

# auto-calculate steps_per_epoch
if total_train_tokens and tokens_per_step > 0:
    steps_per_epoch = total_train_tokens // tokens_per_step
else:
    steps_per_epoch = None

max_steps = 0 # will be calculated by train.py
warmup_steps = 20
# -------------------------

# logging
log_interval = 10
eval_interval = 100 # evaluate every 100 steps

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

# Instruction Dataset (Alpaca)
dataset_name = "tatsu-lab"
data_files_pattern_train = None # Not needed for Alpaca
data_files_pattern_val = None   # Not needed for Alpaca

dataset_split_train = "train"
dataset_split_val = "validation " # Alpaca doesn't have a 'validation' split, manual split
streaming = False

output_dir = "checkpoints"