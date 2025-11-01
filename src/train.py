import torch
import torch.optim as optim
import wandb
import time
from contextlib import nullcontext
import math
import torch.nn.functional as F
import os

import src.config as config
from src.model import create_mamba_model
from src.dataset import get_tokenizer, get_dataloader # Updated dataset functions
from src.optimizer import create_hybrid_optimizer # Creates the list [Muon, AdamW]
from src.muon_optimizer import Muon

def get_lr(it, max_steps, warmup_steps):
    min_lr_factor = config.learning_rate_adamw / 10 / config.learning_rate_adamw
    max_lr_factor = 1.0

    if it < warmup_steps:
        return max_lr_factor * (it / warmup_steps)  # Factor scales from 0.0 -> 1.0

    if it > max_steps:
        return min_lr_factor  # Factor stays at 0.1

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff goes from 1.0 -> 0.0
    return min_lr_factor + coeff * (max_lr_factor - min_lr_factor)


def train():
    max_steps = config.max_steps
    warmup_steps = config.warmup_steps

    if config.train_by_epochs:
        if not config.steps_per_epoch:
            raise ValueError(
                "config.steps_per_epoch must be set when config.train_by_epochs is True."
            )

        # Override max_steps and warmup_steps based on epochs
        original_max_steps = max_steps
        max_steps = config.num_epochs * config.steps_per_epoch

        # Optional: Scale warmup steps proportionally
        if original_max_steps > 0:  # Avoid division by zero
            warmup_ratio = warmup_steps / original_max_steps
        else:
            # Fallback: Warm up for 1% of total steps if original_max_steps was 0
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
        "data_files_train": config.data_files_pattern_train,

        "train_by_epochs": config.train_by_epochs,
        "num_epochs": config.num_epochs if config.train_by_epochs else None,
        "steps_per_epoch": config.steps_per_epoch if config.train_by_epochs else None,
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

    print("Loading tokenizer and creating data iterators...")
    tokenizer = get_tokenizer()
    train_data_iter = get_dataloader(split="train")
    val_data_iter = get_dataloader(split="validation")
    print("Data iterators created.")

    print("Creating Mamba model...")
    model = create_mamba_model()
    model.to(config.device)
    print("Model created.")

    print("Creating hybrid optimizer...")
    current_step = 0
    current_epoch = 0
    optimizers = create_hybrid_optimizer(model)
    print("Optimizer created.")

    best_val_loss = float('inf')

    if config.resume_from_checkpoint:
        if os.path.exists(config.latest_checkpoint_path):
            print(f"--- Resuming from checkpoint: {config.latest_checkpoint_path} ---")
            checkpoint = torch.load(config.latest_checkpoint_path, map_location=config.device)

            model.load_state_dict(checkpoint['model'])
            optimizers[0].load_state_dict(checkpoint['optimizer_muon'])
            optimizers[1].load_state_dict(checkpoint['optimizer_adamw'])

            current_step = checkpoint['step']
            current_epoch = checkpoint.get('epoch', 0)  # Use .get for backward compatibility
            best_val_loss = checkpoint['best_val_loss']

            print(f"Resumed from step {current_step} (Epoch {current_epoch}) with best val loss {best_val_loss:.4f}")
        else:
            print(
                f"Warning: resume_from_checkpoint is True but checkpoint not found at {config.latest_checkpoint_path}. Starting from scratch.")
    else:
        print("--- Starting training from scratch ---")

    print("--- Starting Training ---")
    start_time = time.time()

    while current_step < max_steps:

        lr_factor = get_lr(current_step, max_steps, warmup_steps) if warmup_steps > 0 else 1.0

        for opt in optimizers:
            base_lr = config.learning_rate_muon if isinstance(opt, Muon) else config.learning_rate_adamw
            for param_group in opt.param_groups:
                param_group['lr'] = base_lr * lr_factor

        if current_step % config.eval_interval == 0 and current_step > 0:
            avg_val_loss = run_validation(model, val_data_iter, ctx, current_step)
            val_data_iter = get_dataloader(split="validation")

            checkpoint = {
                'model': model.state_dict(),
                'optimizer_muon': optimizers[0].state_dict(),
                'optimizer_adamw': optimizers[1].state_dict(),
                'step': current_step,
                'epoch': current_epoch,
                'best_val_loss': best_val_loss,  # Store the *current* best loss
                'config': hyperparameters
            }

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint['best_val_loss'] = best_val_loss
                print(f"  New best val loss: {best_val_loss:.4f}. Saving BEST checkpoint...")

                torch.save(checkpoint, config.best_checkpoint_path)
                print(f"  Best checkpoint saved to {config.best_checkpoint_path}")

            # --- ALWAYS save the "latest" checkpoint for resuming ---
            print(f"  Saving 'latest' checkpoint for resuming...")
            torch.save(checkpoint, config.latest_checkpoint_path)
            print(f"  Resume checkpoint saved to {config.latest_checkpoint_path}")

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        accumulated_loss = 0.0
        for micro_step in range(config.gradient_accumulation_steps):
            try:
                input_ids, labels = next(train_data_iter)
            except StopIteration:
                print(f"Epoch {current_epoch} finished at step {current_step}. Resetting training data iterator.")
                current_epoch += 1
                train_data_iter = get_dataloader(split="train")
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
                "train/step": current_step,
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

    print(f"--- Training Complete (Reached step {current_step}) ---")  # MODIFIED
    wandb.finish()


@torch.no_grad()
def run_validation(model, val_data_iter, ctx, train_step):
    print(f"\nRunning validation at step {train_step}")
    model.eval() # Set model to evaluation mode
    total_val_loss = 0.0
    val_steps = 0

    max_val_steps = config.limit_val_batches if config.limit_val_batches else 50 # Default to 50 batches

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
    print(f"Validation complete ({val_steps} batches in {val_time:.2f}s). Avg Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.4f}")
    wandb.log({
        "val/loss": avg_val_loss,
        "val/perplexity": perplexity,
        # "train/step": train_step
    }, step=train_step)

    model.train() # IMPORTANT: Set model back to training mode
    return avg_val_loss


if __name__ == "__main__":
    train()