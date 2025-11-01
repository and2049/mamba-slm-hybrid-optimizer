import torch
import sys
import os
from contextlib import nullcontext

try:
    import src.finetune_config as config
except ImportError:
    print("Error: Could not import finetune_config.py.")
    print("Please ensure this script is run from the root of the project directory.")
    sys.exit(1)

from src.model import create_mamba_model
from src.dataset import get_tokenizer


def chat():
    torch.manual_seed(8647)
    torch.cuda.manual_seed(8647)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=pt_dtype)

    print(f"Running on device: {config.device} with dtype: {config.dtype}")

    print("Loading tokenizer...")
    tokenizer = get_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("Tokenizer pad_token_id set to eos_token_id.")

    print("Creating Mamba model...")
    model = create_mamba_model()
    model.to(config.device)

    checkpoint_path = os.path.join(config.output_dir, "finetuned.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Fine-tuned checkpoint not found at {checkpoint_path}")
        print("Please run finetune.py first to create a 'finetuned.pt' file.")
        sys.exit(1)

    print(f"Loading fine-tuned checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    try:
        model.load_state_dict(checkpoint['model'], strict=True)
        print("Model state loaded successfully.")
    except Exception as e:
        print(f"Error loading model state: {e}")
        print("Ensure finetune_config.py matches the model that was trained.")
        sys.exit(1)

    model.eval()
    print("Model set to evaluation mode.")

    # --- 5. Start Chat Loop ---
    print("\n--- Mamba Chat ---")
    print("Type your instruction and press Enter.")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        try:
            prompt_text = input("\nYou: ")
            if prompt_text.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break


            formatted_prompt = f"Instruction:\n{prompt_text}\n\nResponse:\n"

            input_ids = tokenizer.encode(
                formatted_prompt,
                return_tensors='pt'
            ).to(config.device)

            print("Assistant: ", end="", flush=True)  # Print prefix immediately

            with torch.no_grad(), ctx:
                generated_tokens = model.generate(
                    input_ids=input_ids,
                    max_length=input_ids.shape[1] + 256,  # max new tokens
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=0.7,  # controls randomness
                    top_k=50  # selects from top 50 most likely tokens
                )


            response_tokens = generated_tokens[0][input_ids.shape[1]:]
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

            print(response_text)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break


if __name__ == "__main__":
    chat()