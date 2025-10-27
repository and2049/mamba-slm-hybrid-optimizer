import torch.optim as optim
import src.config as config
from muon_optimizer import Muon


def create_hybrid_optimizer(model):
    muon_params = []
    adam_params = []

    adam_keywords = [
        "embedding",  # Token embeddings
        "bias",  # All bias terms
        "norm",  # All LayerNorm weights and biases
        "A_log",  # Mamba's 'A' parameter (SSM specific)
        "D",  # Mamba's 'D' parameter (SSM specific)
        "dt_proj",  # Mamba's 'delta' projection layer
        "head",  # The final classifier head (lm_head)
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check 1: Is it a 2D matrix suitable for Muon?
        is_2d_matrix = param.ndim == 2

        # Check 2: Is it on the "Adam-only" safe-list?
        is_adam_only = any(keyword in name for keyword in adam_keywords)

        if is_2d_matrix and not is_adam_only:
            # This is a 2D weight matrix we want to train with Muon
            muon_params.append(param)
            print(f"[Muon]   Assigning: {name} (shape: {param.shape})")
        else:
            # This is an embedding, bias, norm, or SSM param
            adam_params.append(param)
            print(f"[AdamW]  Assigning: {name} (shape: {param.shape})")

    # 3. Instantiate both optimizers with their respective learning rates
    optimizer_muon = Muon(muon_params, lr=config.learning_rate_muon)

    optimizer_adamw = optim.AdamW(
        adam_params,
        lr=config.learning_rate_adamw,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay
    )

    print("------------------------------------------")
    print(f"Created Muon optimizer for {len(muon_params)} parameter groups.")
    print(f"Created AdamW optimizer for {len(adam_params)} parameter groups.")

    #return [optimizer_muon, optimizer_adamw]