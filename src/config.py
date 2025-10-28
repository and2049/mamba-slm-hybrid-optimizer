import torch

wandb_project_name = 'mamba-slm-hybrid-optimizer'

# model config
d_model = 768
n_layer = 24
vocab_size = 50277  # Vocabulary size (GPT-2 tokenizer)
pad_vocab_size_multiple = 8


# training config
limit_train_batches = 100  # (e.g., 100 for a smoke test)
limit_val_batches = 20    # (e.g., 20 for a smoke test)

max_seq_len = 1024
learning_rate_muon = 1e-3   # Muon's LR is typically higher
learning_rate_adamw = 1e-4  # AdamW's LR is typically lower
weight_decay = 0.01

gradient_accumulation_steps = 32  # increase if less vram
micro_batch_size = 1

# training duration
max_steps = 10000         # Total training steps (e.g., 5000 for a test run)
warmup_steps = 100        # Linear warmup steps

# logging
log_interval = 10         # Log training loss every N steps
eval_interval = 100       # Run validation every N steps

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

# streaming-capable dataset from Hugging Face
dataset_name = "HuggingFaceFW/fineweb"
data_files_pattern_train = "sample/10BT/*.parquet"
data_files_pattern_val = None\

dataset_split_train = "train"
dataset_split_val = "validation"
streaming = True # set to True to handle 32GB RAM limit