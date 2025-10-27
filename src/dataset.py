import torch
from transformers import AutoTokenizer
import numpy as np
import os
import src.config as config

DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.bin")
VAL_FILE = os.path.join(DATA_DIR, "val.bin")
DTYPE = np.uint16 # must match dtype used in preprocess.py
TOKENIZER_NAME = "gpt2"

class PrecomputedDataloader:
    def __init__(self, split="train"):
        self.split = split
        self.data_file = TRAIN_FILE if split == "train" else VAL_FILE

        if not os.path.exists(self.data_file):
            print(f"Error: Data file not found at {self.data_file}")
            print("Please run `python preprocess.py` first.")
            exit(1)

        print(f"Initializing dataloader from: {self.data_file}")

        # use numpy.memmap to open the file without loading it into RAM
        self.memmap = np.memmap(self.data_file, dtype=DTYPE, mode="r")
        self.num_tokens = self.memmap.shape[0]
        self.num_sequences = self.num_tokens // config.max_seq_len

        print(f"  Total sequences: {self.num_sequences:,}")

    def __len__(self):
        return self.num_sequences // config.micro_batch_size

    def __iter__(self):
        # create array of indices and shuffle it for training
        indices = np.arange(self.num_sequences)
        if self.split == "train":
            np.random.shuffle(indices)

        for i in range(0, len(indices), config.micro_batch_size):
            batch_indices = indices[i: i + config.micro_batch_size]

            # drops partial batches
            if len(batch_indices) < config.micro_batch_size:
                continue

            batch_x, batch_y = [], []
            for seq_idx in batch_indices:
                start = seq_idx * config.max_seq_len
                end = start + config.max_seq_len

                # get sequence chunk from the memory-mapped file
                x = torch.from_numpy(self.memmap[start:end].astype(np.int64))
                start_x = seq_idx * config.max_seq_len
                end_x = start_x + config.max_seq_len

                # randomly pick a starting point in the file
                idx = torch.randint(len(self.memmap) - config.max_seq_len, (1,)).item()

                # read max_seq_len + 1 tokens from that point
                chunk = torch.from_numpy(
                    self.memmap[idx: idx + config.max_seq_len + 1].astype(np.int64)
                )

                x = chunk[:-1]  # The first L tokens
                y = chunk[1:]  # The last L tokens (shifted)

                batch_x.append(x)
                batch_y.append(y)

            X = torch.stack(batch_x)
            Y = torch.stack(batch_y)

            yield X, Y


def get_dataloader(split="train"):
    return iter(PrecomputedDataloader(split))


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer