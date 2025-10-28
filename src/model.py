# src/model.py

from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
import math # Added for vocab padding calculation

# Try importing config relative to src first, then directly if run as script
try:
    import src.config as config
except ImportError:
    import config # Allows running this file directly for testing

def create_mamba_model():
    """
    Creates the MambaLMHeadModel from the configuration specified in config.py.
    Includes vocabulary padding for hardware efficiency.
    """

    # --- Vocabulary Padding ---
    # Pad vocab_size to the nearest multiple of pad_vocab_size_multiple
    # This optimization improves performance on certain hardware (like NVIDIA Tensor Cores)
    padded_vocab_size = config.vocab_size
    if config.pad_vocab_size_multiple > 1:
        padded_vocab_size = (
            math.ceil(config.vocab_size / config.pad_vocab_size_multiple) *
            config.pad_vocab_size_multiple
        )
        if padded_vocab_size != config.vocab_size:
            print(f"Padding vocab size from {config.vocab_size} to {padded_vocab_size} "
                  f"(multiple of {config.pad_vocab_size_multiple})")

    print(f"Initializing Mamba model with config:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layer: {config.n_layer}")
    print(f"  vocab_size (original): {config.vocab_size}")
    print(f"  vocab_size (padded): {padded_vocab_size}")

    # --- Create Mamba Configuration ---
    mamba_config = MambaConfig(
        d_model=config.d_model,
        n_layer=config.n_layer,
        vocab_size=padded_vocab_size, # Use the padded size
        # Add other Mamba-specific config parameters if needed, e.g.:
        # ssm_cfg=None, # Use default SSM config
        # rms_norm=True,
        # fused_add_norm=True,
        # residual_in_fp32=True,
    )

    # --- Instantiate the Model ---
    # MambaLMHeadModel includes the language model head for next-token prediction
    print("Instantiating MambaLMHeadModel...")
    model = MambaLMHeadModel(mamba_config)

    # --- Print Parameter Count ---
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {params_total/1e6:.2f}M")
    print(f"  Trainable Parameters: {params_trainable/1e6:.2f}M")

    return model

# --- Simple Test ---
# This allows you to run `python src/model.py` to quickly check
# if the model creation works without running the full training.
if __name__ == "__main__":
    print("Running model creation test...")
    try:
        model = create_mamba_model()
        print("\nModel architecture created successfully.")
        # print(model) # Uncomment to see the full model structure if needed
    except Exception as e:
        print(f"\nError creating model: {e}")