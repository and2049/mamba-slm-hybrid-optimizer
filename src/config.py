import torch


max_seq_len = 1024
learning_rate_muon = 1e-3   # Muon's LR is typically higher
learning_rate_adamw = 1e-4  # AdamW's LR is typically lower
weight_decay = 0.01

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

# streaming-capable dataset from Hugging Face
dataset_name = "HuggingFaceFW/fineweb"
data_files_pattern_train = "sample/10BT/*.parquet"
data_files_pattern_val = None\

dataset_split_train = "train"
dataset_split_val = "validation"
streaming = True # set to True to handle 32GB RAM limit