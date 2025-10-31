import torch
import torch.optim as optim
import wandb
import time
from contextlib import nullcontext
import math
import torch.nn.functional as F
import os
from datasets import load_dataset  # Keep this
from torch.utils.data import DataLoader  # Keep this

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


def get_finetune_dataloader(dataset, tokenizer, is_train=True):

    if is_train:
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
        batch_size=config.micro_batch_size,
        num_workers=0
    )

    while True:
        for batch in loader:
            yield batch['input_ids'], batch['labels']

def train():
    max_steps = config.max_steps
    warmup_steps = config.warmup_steps

    if config.train_by_epochs:
        if not config.steps_per_epoch:
            raise ValueError("config.steps_per_epoch must be set when config.train_by_epochs is True.")

        original_max_steps = max_steps
        max_steps = config.num_epochs * config.steps_per_epoch

        if original_max_steps > 0:
            warmup_ratio = warmup_steps / original_max_steps
        else:
            warmup_ratio = 0.01

        warmup_steps = int(max_steps * warmup_ratio)

        print(f"--- Training Mode: EPOCHS ---")
        print(f"  Target Epochs:       {config.num_epochs}")
        print(f"  Steps per Epoch:     {config.steps_per_epoch}")
        print(f"  Calculated max_steps:  {max_steps}")
        print(f"  Calculated warmup_steps: {warmup_steps}")
    else:
        print(f"--- Training Mode: STEPS ---")
        print(f"  Target max_steps:  {max_steps}")
        print(f"  Warmup steps: {warmup_steps}")

    hyperparameters = {
        "d_model": config.d_model,
        "n_layer": config.n_layer,
        "vocab_size": config.vocab_size,
        "pad_vocab_size_multiple": config.pad_vocab_size_multiple,
        "max_seq_len": config.max_seq_len,
        "learning_rate_muon": config.learning_rate_muon,
        "learning_rate_adamw": config.learning_rate_adamw,
        "weight_decay": config.weight_decay,
        "batch_size_effective": config.micro_batch_size * config.gradient_accumulation_steps,
        "micro_batch_size": config.micro_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "dtype": config.dtype,
        "dataset": config.dataset_name,
        "pretrained_checkpoint": config.pretrained_checkpoint_path,
    }
    wandb.init(project=config.wandb_project_name, config=hyperparameters, entity="mamba-slm-hybrid-optimizer")

    os.makedirs(config.output_dir, exist_ok=True)
    torch.manual_seed(8647)
    torch.cuda.manual_seed(8647)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=pt_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))

    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    print(f"Loading '{config.dataset_name}' dataset (streaming={config.streaming})...")
    if config.streaming:
        print("Warning: Alpaca dataset is small. Recommend setting streaming = False in config.")

    full_dataset = load_dataset(config.dataset_name, streaming=config.streaming)

    if config.streaming:
        print("Streaming mode: Creating makeshift train/val splits...")
        train_dataset = full_dataset['train'].take(1000)
        val_dataset = full_dataset['train'].take(100)
    else:
        print("Splitting dataset into train and validation...")
        # Split the 'train' split into 95% train / 5% validation
        split_dataset = full_dataset['train'].train_test_split(test_size=0.05, seed=42)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']

    print("Creating fine-tuning data iterators...")
    train_data_iter = get_finetune_dataloader(train_dataset, tokenizer, is_train=True)
    val_data_iter = get_finetune_dataloader(val_dataset, tokenizer, is_train=False)
    print("Data iterators created.")

    print("Creating Mamba model...")
    model = create_mamba_model()
    model.to(config.device)
    print("Model created.")

    if not os.path.exists(config.pretrained_checkpoint_path):
        raise FileNotFoundError(f"Pre-trained checkpoint not found at {config.pretrained_checkpoint_path}")

    print(f"Loading pre-trained weights from {config.pretrained_checkpoint_path}...")
    checkpoint = torch.load(config.pretrained_checkpoint_path, map_location=config.device)

    model_state_dict = checkpoint['model']
    model.load_state_dict(model_state_dict, strict=True)
    print("Pre-trained weights loaded successfully.")

    print("Creating hybrid optimizer...")
    optimizers = create_hybrid_optimizer(model)
    print("Optimizer created.")

    best_val_loss = float('inf')

    print("--- Starting Fine-Tuning ---")
    start_time = time.time()
    current_step = 0
    current_epoch = 0

    while current_step < max_steps:
        lr_factor = get_lr(current_step, max_steps, warmup_steps) if warmup_steps > 0 else 1.0
        for opt in optimizers:
            base_lr = config.learning_rate_muon if isinstance(opt, Muon) else config.learning_rate_adamw
            for param_group in opt.param_groups:
                param_group['lr'] = base_lr * lr_factor

        if current_step % config.eval_interval == 0 and current_step > 0:
            avg_val_loss = run_validation(model, val_data_iter, ctx, current_step)
            val_data_iter = get_finetune_dataloader(val_dataset, tokenizer, is_train=False)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"  New best val loss: {best_val_loss:.4f}. Saving checkpoint...")
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer_muon': optimizers[0].state_dict(),
                    'optimizer_adamw': optimizers[1].state_dict(),
                    'step': current_step,
                    'epoch': current_epoch,
                    'best_val_loss': best_val_loss,
                    'config': hyperparameters
                }
                best_save_path = os.path.join(config.output_dir, "finetuned.pt")
                torch.save(checkpoint, best_save_path)
                print(f"  Checkpoint saved to {best_save_path}")

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        accumulated_loss = 0.0
        for micro_step in range(config.gradient_accumulation_steps):
            try:
                input_ids, labels = next(train_data_iter)
            except StopIteration:
                print(f"Epoch {current_epoch} finished at step {current_step}. Resetting training data iterator.")
                current_epoch += 1
                train_data_iter = get_finetune_dataloader(train_dataset, tokenizer, is_train=True)
                input_ids, labels = next(train_data_iter)

            input_ids = input_ids.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)

            with ctx:
                logits = model(input_ids).logits
                B, L, V = logits.shape
                logits_for_loss = logits.view(B * L, V)
                labels_for_loss = labels.view(B * L)

                loss = F.cross_entropy(logits_for_loss, labels_for_loss, ignore_index=-100)

            loss = loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

            if config.limit_train_batches and (
                    current_step * config.gradient_accumulation_steps + micro_step + 1) >= config.limit_train_batches:
                break

        scaler.unscale_(optimizers[0])
        scaler.unscale_(optimizers[1])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizers[0])
        scaler.step(optimizers[1])
        scaler.update()

        if current_step % config.log_interval == 0:
            elapsed_time = time.time() - start_time
            current_lr_muon = optimizers[0].param_groups[0]['lr']
            current_lr_adamw = optimizers[1].param_groups[0]['lr']
            tokens_processed = (
                                           current_step + 1) * config.micro_batch_size * config.gradient_accumulation_steps * config.max_seq_len

            print(
                f"Step {current_step:5d} | Epoch {current_epoch} | Loss: {accumulated_loss:.4f} | LR Muon: {current_lr_muon:.2e} | LR AdamW: {current_lr_adamw:.2e} | Tokens: {tokens_processed / 1e6:.1f}M | Time: {elapsed_time:.2f}s")
            wandb.log({
                "train/loss": accumulated_loss,
                "train/epoch": current_epoch,
                "train/lr_muon": current_lr_muon,
                "train/lr_adamw": current_lr_adamw,
                "system/time_seconds": elapsed_time,
                "system/tokens_processed": tokens_processed
            }, step=current_step)

        current_step += 1

        if config.limit_train_batches and current_step >= (
                config.limit_train_batches // config.gradient_accumulation_steps):
            print("Reached training batch limit. Stopping.")
            break

    print(f"--- Fine-Tuning Complete (Reached step {current_step}) ---")
    wandb.finish()


@torch.no_grad()
def run_validation(model, val_data_iter, ctx, train_step):
    print(f"\nRunning validation at step {train_step}")
    model.eval()
    total_val_loss = 0.0
    val_steps = 0

    max_val_steps = config.limit_val_batches if config.limit_val_batches else 50

    start_val_time = time.time()
    for _ in range(max_val_steps):
        try:
            input_ids, labels = next(val_data_iter)
        except StopIteration:
            print("Validation data iterator exhausted early.")
            break

        input_ids = input_ids.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)

        with ctx:
            logits = model(input_ids).logits
            B, L, V = logits.shape
            logits_for_loss = logits.view(B * L, V)
            labels_for_loss = labels.view(B * L)

            loss = F.cross_entropy(logits_for_loss, labels_for_loss, ignore_index=-100)
            total_val_loss += loss.item()

        val_steps += 1

    if val_steps == 0:
        print("Warning: No validation batches were processed.")
        avg_val_loss = float('inf')
        perplexity = float('inf')
    else:
        avg_val_loss = total_val_loss / val_steps
        try:
            perplexity = math.exp(avg_val_loss)
        except OverflowError:
            perplexity = float('inf')

    val_time = time.time() - start_val_time
    print(
        f"Validation complete ({val_steps} batches in {val_time:.2f}s). Avg Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.4f}")
    wandb.log({
        "val/loss": avg_val_loss,
        "val/perplexity": perplexity,
    }, step=train_step)

    model.train()
    return avg_val_loss


if __name__ == "__main__":
    train()