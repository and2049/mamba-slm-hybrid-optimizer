import os
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import src.config as config

TOKENIZER_NAME = "gpt2"
DATASET_NAME = config.dataset_name
TRAIN_SPLIT = config.dataset_split_train
VAL_SPLIT = config.dataset_split_val
MAX_SEQ_LEN = config.max_seq_len
OUTPUT_DIR = "data"
OUTPUT_TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.bin")
OUTPUT_VAL_FILE = os.path.join(OUTPUT_DIR, "val.bin")

dtype = np.uint16

def process_and_save(split_name, output_file):
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping preprocessing for {split_name}.")
        # try to load the existing file to get its size for confirmation
        try:
            existing_memmap = np.memmap(output_file, dtype=dtype, mode='r')
            final_size = existing_memmap.shape[0]
            print(f"  Existing file size: {final_size:,} tokens ({final_size / 1e6:.2f}M)")
            del existing_memmap  # Close the file handle
        except Exception as e:
            print(f"  Warning: Could not read existing file size: {e}")
        return  # skip the rest of the function for this split
    print(f'Loading tokenizer: {TOKENIZER_NAME}')
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: '{config.dataset_name}' (split: {split_name}")
    if split_name == TRAIN_SPLIT:
        dataset = load_dataset(
            config.dataset_name,
            data_files=config.data_files_pattern_train,
            split=config.dataset_split_train,
            streaming=True
        )
    elif split_name == VAL_SPLIT:
        print("Using a fraction of the training data for validation...")
        full_train_dataset = load_dataset(
            config.dataset_name,
            data_files=config.data_files_pattern_train,
            split=config.dataset_split_train,
            streaming=True
        )
        dataset = full_train_dataset.take(100000)

    else:
        raise ValueError(f"Unknown split name: {split_name}")

    temp_file = output_file + ".tmp"
    print(f"Tokenizing and writing to temp file: {temp_file}")
    pbar = tqdm(desc=f"Processing {split_name}", unit=" tokens")

    token_count = 0
    with open(temp_file, "wb") as f:
        for example in dataset:
            text = example["text"]
            if not text:
                continue

            tokenized_output = tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                max_length=config.max_seq_len
            )
            tokens = tokenized_output["input_ids"][0]
            tokens_with_eos = np.append(tokens, tokenizer.eos_token_id)

            f.write(tokens_with_eos.astype(dtype).tobytes())
            token_count += len(tokens_with_eos)
            pbar.update(len(tokens_with_eos))

    pbar.close()

    print(f"\nTotal tokens processed: {token_count}")
    print(f"Creating final chunked file: {output_file}")

    num_sequences = (token_count // (MAX_SEQ_LEN + 1))
    final_size = num_sequences * MAX_SEQ_LEN

    if final_size == 0:
        print(f"Error: Not enough tokens ({token_count}) to create even one sequence of length {MAX_SEQ_LEN}.")
        return

    final_memmap = np.memmap(output_file, dtype=dtype, mode="w+", shape=(final_size,))
    temp_memmap = np.memmap(temp_file, dtype=dtype, mode="r", shape=(token_count,))

    print(f"Writing {num_sequences:,} sequences of length {MAX_SEQ_LEN}")

    write_idx = 0
    for i in tqdm(range(num_sequences)):
        start_idx = i * (MAX_SEQ_LEN + 1)
        end_idx = start_idx + MAX_SEQ_LEN

        chunk = temp_memmap[start_idx: end_idx]

        final_memmap[write_idx: write_idx + MAX_SEQ_LEN] = chunk
        write_idx += MAX_SEQ_LEN

    final_memmap.flush()
    del final_memmap
    del temp_memmap

    # Clean up the temp file
    # os.remove(temp_file) # - results in WinError 32, manually delete

    print(f"Successfully created: {output_file}")
    print(f"Final shape: {final_size} tokens ({final_size / 1e6:.2f}M)")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    process_and_save(TRAIN_SPLIT, OUTPUT_TRAIN_FILE)
    process_and_save(VAL_SPLIT, OUTPUT_VAL_FILE)
    print("\nData preprocessing complete.")
