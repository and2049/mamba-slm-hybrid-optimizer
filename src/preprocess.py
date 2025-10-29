import os
import time
import math
from typing import Optional, Dict, List, Iterator

import numpy as np
from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer
from tqdm import tqdm

try:
    import src.config as config
except ImportError:
    import config

TOKENIZER_NAME = "gpt2"
DATASET_NAME = config.dataset_name
DATA_FILES_TRAIN = config.data_files_pattern_train
DATA_FILES_VAL = config.data_files_pattern_val

TRAIN_SPLIT_NAME = config.dataset_split_train
VAL_SPLIT_NAME = config.dataset_split_val
MAX_SEQ_LEN = config.max_seq_len

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")

OUTPUT_TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.bin")
OUTPUT_VAL_FILE = os.path.join(OUTPUT_DIR, "val.bin")

NUMPY_DTYPE = np.uint16
BYTES_PER_TOKEN = np.dtype(NUMPY_DTYPE).itemsize

def get_tokenizer(tokenizer_name: str = TOKENIZER_NAME) -> AutoTokenizer:
    print(f"Loading tokenizer: '{tokenizer_name}'")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer pad_token to eos_token ({tokenizer.eos_token_id})")
    return tokenizer

def load_streaming_dataset(
    dataset_name: str,
    split_name: str,
    data_files_pattern: Optional[str] = None
) -> IterableDataset:
    print(f"Loading dataset '{dataset_name}' (split: {split_name}, streaming: True)...")
    try:
        dataset = load_dataset(
            dataset_name,
            name=None,
            data_files=data_files_pattern,
            split=split_name,
            streaming=True
        )
        # Peek at the first example to check structure (optional)
        first_example = next(iter(dataset))
        print("Dataset loaded. First example keys:", first_example.keys())
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset name, split, and data_files pattern are correct.")
        exit(1)

def tokenize_and_yield_tokens(dataset: IterableDataset, tokenizer: AutoTokenizer) -> Iterator[np.ndarray]:
    count = 0
    for example in dataset:
        text = example.get("text")
        if text:
            token_ids = tokenizer(text, return_attention_mask=False)["input_ids"]
            if token_ids:
                token_ids.append(tokenizer.eos_token_id)
                yield np.array(token_ids, dtype=NUMPY_DTYPE)
                count += len(token_ids)

def write_tokens_to_tempfile(
    token_iterator: Iterator[np.ndarray],
    temp_file_path: str,
    split_name: str
) -> int:
    total_tokens = 0
    print(f"Tokenizing and writing to temp file: {temp_file_path}")
    with open(temp_file_path, "wb") as f, \
         tqdm(desc=f"Processing {split_name}", unit=" tokens") as pbar:
        for tokens_array in token_iterator:
            f.write(tokens_array.tobytes())
            total_tokens += len(tokens_array)
            pbar.update(len(tokens_array))
    print(f"\nFinished writing temp file. Total tokens processed: {total_tokens:,}")
    return total_tokens

def chunk_tempfile_to_final(
    temp_file_path: str,
    output_file_path: str,
    total_tokens: int,
    max_seq_len: int
):
    print(f"Creating final chunked file: {output_file_path}")

    num_sequences = total_tokens // max_seq_len
    final_size_tokens = num_sequences * max_seq_len

    if final_size_tokens == 0:
        print(f"Error: Not enough tokens ({total_tokens}) to create even one sequence of length {max_seq_len}.")
        try:
            os.remove(temp_file_path)
        except OSError:
            pass
        return

    print(f"Calculated final size: {final_size_tokens:,} tokens ({final_size_tokens * BYTES_PER_TOKEN / (1024**3):.2f} GB)")
    print(f"Number of sequences: {num_sequences:,}")

    try:
        final_memmap = np.memmap(output_file_path, dtype=NUMPY_DTYPE, mode="w+", shape=(final_size_tokens,))
    except Exception as e:
        print(f"Error creating final memmap file at {output_file_path}: {e}")
        return

    try:
        temp_memmap = np.memmap(temp_file_path, dtype=NUMPY_DTYPE, mode="r", shape=(total_tokens,))
    except Exception as e:
        print(f"Error opening temp memmap file at {temp_file_path}: {e}")
        del final_memmap
        return

    print(f"Copying {num_sequences:,} sequences of length {max_seq_len}")

    chunk_size_mem = 1 * (1024**3) # orocess in chunks (e.g., 1GB) to manage RAM for large files
    tokens_per_chunk = chunk_size_mem // BYTES_PER_TOKEN
    sequences_per_chunk = tokens_per_chunk // max_seq_len
    num_chunks = math.ceil(num_sequences / sequences_per_chunk)

    processed_sequences = 0
    with tqdm(total=num_sequences, desc="Chunking data", unit=" sequences") as pbar:
        for chunk_idx in range(num_chunks):
            start_seq = chunk_idx * sequences_per_chunk
            end_seq = min((chunk_idx + 1) * sequences_per_chunk, num_sequences)
            if start_seq >= end_seq:
                break

            start_token_read = start_seq * max_seq_len
            end_token_read = end_seq * max_seq_len

            # Read a large chunk from temp file
            data_chunk = temp_memmap[start_token_read:end_token_read]

            # Write the chunk to the final file
            start_token_write = start_seq * max_seq_len
            end_token_write = start_token_write + len(data_chunk)
            final_memmap[start_token_write:end_token_write] = data_chunk

            sequences_in_chunk = end_seq - start_seq
            processed_sequences += sequences_in_chunk
            pbar.update(sequences_in_chunk)


    final_memmap.flush()
    del final_memmap
    del temp_memmap

    print(f"Successfully created final file: {output_file_path}")
    print(f"Final shape: {final_size_tokens:,} tokens ({final_size_tokens / 1e6:.2f}M)")

    print(f"Attempting to clean up temporary file: {temp_file_path}")
    time.sleep(1)
    try:
        os.remove(temp_file_path)
        print("Successfully removed temporary file.")
    except PermissionError as e:
        if hasattr(e, 'winerror') and e.winerror == 32: # Check WinError 32
            print(f"Warning: Could not remove {temp_file_path} (still in use?). Manual cleanup might be needed.")
        else:
            print(f"Warning: PermissionError removing temp file: {e}")
    except Exception as e:
        print(f"Warning: Error removing temp file: {e}")


def process_split(split_name: str, output_file: str, data_files_pattern: Optional[str] = None):
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping.")
        try:
            existing_memmap = np.memmap(output_file, dtype=NUMPY_DTYPE, mode='r')
            print(f"  Existing file size: {existing_memmap.shape[0]:,} tokens")
            del existing_memmap
        except Exception as e:
            print(f"  Warning: Could not read existing file size: {e}")
        return

    tokenizer = get_tokenizer()

    if split_name == VAL_SPLIT_NAME and not data_files_pattern:
        print("Creating validation split by taking subset of training data stream...")
        full_train_dataset = load_streaming_dataset(DATASET_NAME, TRAIN_SPLIT_NAME, DATA_FILES_TRAIN)
        val_examples_count = 100000
        dataset = full_train_dataset.take(val_examples_count)
        print(f"Using first {val_examples_count} examples from train stream for validation.")
    else:
        dataset = load_streaming_dataset(DATASET_NAME, split_name, data_files_pattern)

    temp_file = output_file + ".tmp"
    token_iterator = tokenize_and_yield_tokens(dataset, tokenizer)
    total_tokens = write_tokens_to_tempfile(token_iterator, temp_file, split_name)

    if total_tokens > 0:
        chunk_tempfile_to_final(temp_file, output_file, total_tokens, MAX_SEQ_LEN)
    else:
        print(f"No tokens processed for split {split_name}. Final file not created.")
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass


if __name__ == "__main__":
    start_main_time = time.time()
    print("Starting data preprocessing...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n--- Processing Training Split ---")
    process_split(TRAIN_SPLIT_NAME, OUTPUT_TRAIN_FILE, data_files_pattern=DATA_FILES_TRAIN)

    print("\n--- Processing Validation Split ---")
    process_split(VAL_SPLIT_NAME, OUTPUT_VAL_FILE, data_files_pattern=DATA_FILES_VAL)

    end_main_time = time.time()
    print(f"\nData preprocessing check complete. Total time: {end_main_time - start_main_time:.2f} seconds.")