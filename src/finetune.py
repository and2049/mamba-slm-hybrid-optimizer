import torch
import torch.optim as optim
import wandb
import time
from contextlib import nullcontext
import math
import torch.nn.functional as F
import os
from datasets import load_dataset
from torch.utils.data import DataLoader

import src.finetune_config as config

from src.model import create_mamba_model
from src.dataset import get_tokenizer
from src.optimizer import create_hybrid_optimizer
from src.muon_optimizer import Muon

def get_lr(it, max_steps, warmup_steps):
    min_lr_factor = config.learning_rate_adamw / 10 / config.learning_rate_adamw
    max_lr_factor = 1.0
    if it < warmup_steps:
        return max_lr_factor * (it / warmup_steps)
    if it > max_steps:
        return min_lr_factor
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr_factor + coeff * (max_lr_factor - min_lr_factor)


def get_finetune_dataloader(split, tokenizer):
    print(f"Loading '{config.dataset_name}' dataset (split: {split}, streaming: {config.streaming})...")

    dataset = load_dataset(
        config.dataset_name,
        split=split,
        streaming=config.streaming
    )

    if split == config.dataset_split_train:
        dataset = dataset.shuffle(buffer_size=10000, seed=42)

    def format_prompt(example):
        if example.get("input", ""):
            prompt = f"Instruction:\n{example['instruction']}\n\nInput:\n{example['input']}\n\nResponse:\n{example['output']}"
        else:
            prompt = f"Instruction:\n{example['instruction']}\n\nResponse:\n{example['output']}"
        return prompt

    def format_prompt_for_loss(example):
        if example.get("input", ""):
            prompt = f"Instruction:\n{example['instruction']}\n\nInput:\n{example['input']}\n\nResponse:\n"
        else:
            prompt = f"Instruction:\n{example['instruction']}\n\nResponse:\n"
        return prompt

    def process_example(example):
        full_text = format_prompt(example) + tokenizer.eos_token
        prompt_text = format_prompt_for_loss(example)

        full_tokenized = tokenizer(
            full_text,
            max_length=config.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        prompt_tokenized = tokenizer(
            prompt_text,
            max_length=config.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        input_ids = full_tokenized['input_ids']
        labels = list(input_ids)

        prompt_len = len(prompt_tokenized['input_ids'])

        for i in range(prompt_len):
            if i < len(labels):
                labels[i] = -100

        pad_len = config.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        input_ids = input_ids[:config.max_seq_len]
        labels = labels[:config.max_seq_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    dataset = dataset.map(process_example)
    dataset = dataset.with_format("torch")

    loader = DataLoader(
        dataset,
        batch_size=config.micro_batch_size,  # this is our micro_batch_size
        num_workers=0  # simpler for streaming
    )

    while True:
        for batch in loader:
            yield batch['input_ids'], batch['labels']

