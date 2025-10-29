# Mamba Small Language Model with Hybrid Optimizer

This project implements and trains a small language model using the Mamba architecture, specifically designed as an state space model (SSM) for efficient sequence modeling. It employs a hybrid optimization strategy combining Muon (for matrix parameters) and AdamW (for embeddings and normalization parameters) to achieve optimal performance. The model is trained on the FineWeb dataset and demonstrates competitive results for its size.

## How It Works: Mamba Architecture and Hybrid Optimization

### Mamba State Space Model
Mamba is a powerful alternative to transformers for sequence modeling. Instead of attention mechanisms, it uses structured state space models (SSMs) that can efficiently process long sequences with linear computation complexity. Key advantages include:

- **Linear Scaling**: Complexity O(N) instead of O(N²) like transformers
- **Hardware Efficient**: Optimized for GPUs with selective state computations
- **Strong Performance**: Competitive with or better than transformers on language tasks

The architecture consists of stacked Mamba blocks, each containing SSM layers with gating mechanisms and residual connections.

### Hybrid Optimizer Strategy
Traditional optimizers struggle with the different parameter types in modern architectures. This project uses a sophisticated hybrid approach:

- **Muon Optimizer**: Applied to 2D weight matrices (most model parameters). Muon performs Newton-Schulz orthogonalization to accelerate convergence and stabilize training. Based on the Muon algorithm that orthogonalizes gradient updates.
- **AdamW Optimizer**: Used for embeddings, biases, normalization parameters, and SSM-specific parameters like A_log, D, and dt_proj. AdamW provides stable optimization for these sensitive parameters.

This hybrid setup allows each parameter group to be optimized with its most suitable algorithm, leading to faster convergence and better final performance.

## Data and Training

### Dataset Preparation
The model is trained on the FineWeb dataset, a high-quality web crawl dataset that provides diverse and comprehensive training data:

- **Source**: HuggingFaceFW/fineweb
- **Streaming**: Uses streaming to handle large datasets without loading everything into memory
- **Preprocessing**: Text is tokenized using GPT-2 tokenizer, then saved as precomputed binary files
- **Sequence Length**: 1024 tokens per sequence

Preprocessing converts raw text into tokenized sequences, stored in efficient binary format for streaming during training.

### Training Configuration
Key hyperparameters optimized for performance:

- **Model Size**: 768 dimensions, 24 layers, ~1.2B parameters
- **Batch Configuration**: Micro batch size of 1 with gradient accumulation (32 steps)
- **Learning Rates**: Muon at 1e-3, AdamW at 1e-4 with warmup and cosine decay
- **Regularization**: Weight decay 0.01, gradient clipping at 1.0
- **Mixed Precision**: bfloat16 for memory efficiency
- **Training Duration**: 10,000 steps with linear warmup (100 steps)

## Project Structure
```
mamba-slm-hybrid-optimizer/
│
├── data/                     # Stores preprocessed training data
├── src/                      # All Python source code
│   ├── config.py             # Central configuration for all parameters
│   ├── model.py              # Mamba model creation and architecture
│   ├── muon_optimizer.py     # Muon optimizer implementation
│   ├── optimizer.py          # Hybrid optimizer setup logic
│   ├── dataset.py            # Data loading and preprocessing utilities
│   ├── preprocess.py         # Script to preprocess raw data into binary format
│   ├── train.py              # Main training script
│
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## How to Run the Project

### Setup Environment
Install the required dependencies:

```python
pip install -r requirements.txt
```

### Prepare Data
Download and preprocess the FineWeb dataset. This creates binary files in the data directory.

```python
python src/preprocess.py
```

This step converts raw text data into tokenized sequences suitable for efficient training.

### Train the Model
Start the training process. The script handles hybrid optimization and logs progress.

```python
python src/train.py
```

Training includes:
- Automatic optimizer assignment (Muon/AdamW)
- Mixed precision training
- Gradient accumulation for larger effective batches
- Weights & Biases logging for monitoring
- Regular validation and checkpointing



## Citations

Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv preprint arXiv:2312.00752.

Jordan, K. (2023). The Muon Optimizer. https://kellerjordan.github.io/posts/muon/

Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1412.6980.

Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization. arXiv preprint arXiv:1711.05101.
