# Mamba Small Language Model with Hybrid Optimizer

This project implements and trains a small language model using the Mamba architecture, specifically designed as a state space model (SSM) for efficient sequence modeling. It employs a hybrid optimization strategy combining Muon (for matrix parameters) and AdamW (for embeddings and normalization parameters) to achieve optimal performance.

The project is structured in two main phases:

1.  **Pre-training**: Training a base model on a massive web dataset (`HuggingFaceFW/fineweb`) to learn general language and world knowledge.
2.  **Supervised Fine-Tuning (SFT)**: Taking the pre-trained base model and instruction-tuning it on a specialized dataset (`HuggingFaceH4/alpaca`) to make it a helpful, instruction-following assistant.

## How It Works: Mamba Architecture and Hybrid Optimization

### Mamba State Space Model

Mamba is a powerful alternative to transformers for sequence modeling. Instead of attention mechanisms, it uses structured state space models (SSMs) that can efficiently process long sequences with linear computation complexity. Key advantages include:

  - **Linear Scaling**: Complexity $O(N)$ instead of $O(N^2)$ like transformers
  - **Hardware Efficient**: Optimized for GPUs with selective state computations
  - **Strong Performance**: Competitive with or better than transformers on language tasks

The architecture consists of stacked Mamba blocks, each containing SSM layers with gating mechanisms and residual connections.

### Hybrid Optimizer Strategy

Traditional optimizers struggle with the different parameter types in modern architectures. This project uses a sophisticated hybrid approach:

  - **Muon Optimizer**: Applied to 2D weight matrices (most model parameters). Muon performs Newton-Schulz orthogonalization to accelerate convergence and stabilize training. Based on the Muon algorithm that orthogonalizes gradient updates.
  - **AdamW Optimizer**: Used for embeddings, biases, normalization parameters, and SSM-specific parameters like `A_log`, `D`, and `dt_proj`. AdamW provides stable optimization for these sensitive parameters.

This hybrid setup allows each parameter group to be optimized with its most suitable algorithm, leading to faster convergence and better final performance.

## Training Process: From Base Model to Chat Assistant

### Phase 1: Pre-training (`train.py`)

The first phase creates the base model by training it on a massive, general-purpose dataset.

  - **Script**: `src/train.py`
  - **Config**: `src/config.py`
  - **Dataset**: `HuggingFaceFW/fineweb`. The data is loaded using Hugging Face's `streaming=True` mode, which avoids downloading the entire multi-terabyte dataset.
  - **Goal**: To learn the fundamentals of language, grammar, and world knowledge.
  - **Output**: A base model checkpoint, saved to `checkpoints/best.pt`.

### Phase 2: Supervised Fine-Tuning (`finetune.py`)

The second phase teaches the pre-trained base model how to follow instructions and respond helpfully.

  - **Script**: `src/finetune.py`
  - **Config**: `src/finetune_config.py`
  - **Dataset**: `HuggingFaceH4/alpaca`. This is a popular dataset of instructions and high-quality responses.
  - **Goal**: To align the model's behavior with user expectations for a chat assistant.
  - **Key Logic**:
    1.  Loads the `best.pt` checkpoint from Phase 1.
    2.  Uses much **lower learning rates** for gentle "nudging" of the weights.
    3.  Applies **loss masking**: The model is only trained to predict the *assistant's response*, and the user's prompt tokens are ignored in the loss calculation.
  - **Output**: An instruction-tuned model, saved to `checkpoints/finetuned.pt`.

## Flexible Training Duration

Both training and fine-tuning can be configured to run for a set number of steps or a set number of epochs. This is controlled in the config files:

  - **`train_by_epochs = False`**: The script will run for the exact number of steps specified in `max_steps`.
  - **`train_by_epochs = True`**: The script will run for `num_epochs`. This requires you to provide an *estimate* of the dataset size in `total_train_tokens` so the script can automatically calculate the total `max_steps` for the LR scheduler.

## Project Structure

```
mamba-slm-hybrid-optimizer/
│
├── src/
│   ├── config.py             # Central config for pre-training (Phase 1)
│   ├── finetune_config.py    # Central config for fine-tuning (Phase 2)
│   ├── model.py              # Mamba model creation and architecture
│   ├── muon_optimizer.py     # Muon optimizer implementation
│   ├── optimizer.py          # Hybrid optimizer setup logic
│   ├── dataset.py            # Data loading utilities (e.g., get_tokenizer)
│   ├── train.py              # Main pre-training script (Phase 1)
│   ├── finetune.py           # Main fine-tuning script (Phase 2)
│
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## How to Run the Project

### Step 1: Setup Environment

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Step 2: Configure Pre-training (Optional)

Review and edit `src/config.py` to set your parameters. Key settings to check:

  - `wandb_project_name`: Your Weights & Biases project.
  - `dataset_name`: The dataset to stream (default: `HuggingFaceFW/fineweb`).
  - `train_by_epochs`: Set to `True` or `False`.
  - `total_train_tokens`: **Required if `train_by_epochs=True`**. (e.g., `10_000_000_000` for 10B tokens).

### Step 3: Run Pre-training (Phase 1)

Start the pre-training process. This will stream the `fineweb` dataset, train the model, and save the best checkpoint to `checkpoints/best.pt`.

```bash
python -m src.train
```

This step is compute-intensive and designed to run for a long time (days or weeks) on powerful GPUs.

### Step 4: Configure Fine-Tuning (Optional)

Once pre-training is complete, review and edit `src/finetune_config.py`.

  - `pretrained_checkpoint_path`: Ensure this points to your saved checkpoint (e.g., `checkpoints/best.pt`).
  - `dataset_name`: The instruction dataset (default: `tatsu-lab/alpaca`).
  - `num_epochs`: 1-3 epochs is typical for fine-tuning.
  - `total_train_tokens`: Update this to reflect the size of the *fine-tuning* dataset (e.g., \~10M for Alpaca).
  - `learning_rate_muon` & `learning_rate_adamw`: These should be much lower than in pre-training.

### Step 5: Run Fine-Tuning (Phase 2)

Start the fine-tuning process. This loads your base model, streams the `alpaca` dataset, and saves the final instruction-tuned model to `checkpoints/finetuned.pt`.

```bash
python -m src.finetune
```

This step is much faster than pre-training and can be completed in a few hours on a single GPU.

## Citations

Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv preprint arXiv:2312.00752.

Jordan, K. (2023). The Muon Optimizer. [https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/)

Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1412.6980.

Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization. arXiv preprint arXiv:1711.05101.