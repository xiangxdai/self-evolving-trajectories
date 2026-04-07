"""
Generic fixed-length character-level preprocessing for countdown-style tasks.

This script builds the shared vocabulary, serializes each example as
[quiz_padded][response_padded], writes flattened uint16 train/val binaries, and
stores metadata needed by AR/Tom-CAT evaluation and training entrypoints.
"""

import argparse
import json
import os
import pickle
import random
import string

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Generic data preparation for fixed-length countdown-style tasks')

    # Input/output paths.
    parser.add_argument('--data_path', type=str, default='data/cd4_train.jsonl', help='Path to cd5 JSONL file')
    parser.add_argument('--out_dir', type=str, default='data/cd/cd4/k1', help='Output directory for processed data')

    # Dataset schema.
    parser.add_argument('--input_key', type=str, default='input', help='JSON key for the prompt/quiz')
    parser.add_argument('--output_key', type=str, default='output', help='JSON key for the completion/response')
    parser.add_argument('--meta_name', type=str, default='meta.pkl', help='Name of the output metadata file')

    # Train/validation split.
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of data to use for validation (e.g., 0.1 for 10%)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling and splitting')

    # Vocabulary control.
    parser.add_argument('--custom_vocab', type=str, default='', help='Comma-separated custom characters. If empty, auto-scans the dataset.')

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # 1. Read the JSONL source file.
    print(f"Loading data from {args.data_path}...")
    data = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(data)} samples.")

    # 2. Build the vocabulary.
    special_tokens = ["<PAD>", "<SEP>", "<EOS>", "<MASK>", "$"]

    # Global base character set shared across common reasoning tasks.
    global_base_chars = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        ",", "+", "-", "/", "=", "*",
    ] + list(string.ascii_lowercase)
    
    if args.custom_vocab:
        print("Using custom vocabulary...")
        base_chars = [c.strip() for c in args.custom_vocab.split(',') if c.strip()]
    else:
        print("Using Global Unified Vocabulary...")
        base_chars = global_base_chars.copy()

        # Scan the dataset to append any unseen characters beyond the shared base set.
        all_chars = set()
        for sample in data:
            all_chars.update(list(str(sample.get(args.input_key, ''))))
            all_chars.update(list(str(sample.get(args.output_key, ''))))
        
        unseen_chars = set(all_chars) - set(base_chars) - set(special_tokens)
        if unseen_chars:
            print(f"⚠️ Notice: Found new characters not in global vocab: {unseen_chars}")
            base_chars.extend(sorted(list(unseen_chars)))

    # Keep special tokens unique and place them at the end of the vocabulary.
    chars = [c for c in base_chars if c not in special_tokens] + special_tokens
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    if vocab_size >= 65536:
        raise ValueError(f"vocab_size={vocab_size} exceeds uint16 capacity")
    print(f"Vocab size: {vocab_size}")

    def encode(s):
        return [stoi[c] for c in str(s)]

    # 3. Compute the maximum raw input/output lengths.
    max_quiz_len = 0
    max_response_len = 0
    for sample in data:
        quiz = str(sample.get(args.input_key, ''))
        response = str(sample.get(args.output_key, ''))
        max_quiz_len = max(max_quiz_len, len(quiz))
        max_response_len = max(max_response_len, len(response))

    quiz_size = max_quiz_len + 1      # +1 for <SEP>
    response_size = max_response_len + 1  # +1 for <EOS>
    data_size = quiz_size + response_size

    print(f"max_quiz_len={max_quiz_len}, max_response_len={max_response_len}")
    print(f"quiz_size={quiz_size}, response_size={response_size}, data_size={data_size}")

    # 4. Shuffle and split into train/validation sets.
    random.shuffle(data)
    num_val = int(len(data) * args.val_ratio)
    val_samples = data[:num_val]
    train_samples = data[num_val:]
    print(f"Split: {len(train_samples)} train samples, {len(val_samples)} val samples.")

    # 5. Encode samples into the fixed-length serialization:
    #    [quiz chars + PAD ... + SEP][response chars + PAD ... + EOS]
    def process_samples(samples, dataset_name):
        processed_seqs = []
        for idx, sample in enumerate(samples):
            quiz = str(sample.get(args.input_key, ''))
            response = str(sample.get(args.output_key, ''))
            
            quiz_encoded = encode(quiz)
            response_encoded = encode(response)

            # Pad the quiz and response to their dataset-wide maximum lengths.
            quiz_padded = quiz_encoded + [stoi["<PAD>"]] * (max_quiz_len - len(quiz_encoded)) + [stoi["<SEP>"]]
            response_padded = response_encoded + [stoi["<PAD>"]] * (max_response_len - len(response_encoded)) + [stoi["<EOS>"]]
            
            seq = quiz_padded + response_padded
            
            if len(seq) != data_size:
                print(f"[{dataset_name}] Skipping invalid sequence at index {idx}: seq_len={len(seq)}, expected={data_size}")
                continue
                
            processed_seqs.extend(seq)
        return processed_seqs

    train_data = process_samples(train_samples, "Train")
    val_data = process_samples(val_samples, "Val")

    print(f"Raw train tokens: {len(train_data)}")
    print(f"Raw val tokens: {len(val_data)}")

    # Print a few decoded examples to verify the packing protocol.
    print("\n" + "="*60)
    print("VERIFICATION: Checking a few processed training samples...")
    print("="*60)
    num_examples_to_print = 3
    if len(train_data) >= data_size * num_examples_to_print:
        for i in range(num_examples_to_print):
            start_idx = i * data_size
            end_idx = start_idx + data_size
            sample_seq = train_data[start_idx:end_idx]

            decoded_seq = [itos[token_id] for token_id in sample_seq]

            quiz_part = "".join(decoded_seq[:quiz_size])
            resp_part = "".join(decoded_seq[quiz_size:])

            print(f"--- Example {i+1} ---")
            print(f"Padded Quiz     (len={len(quiz_part)}): {quiz_part}")
            print(f"Padded Response (len={len(resp_part)}): {resp_part}")
            print()
    else:
        print("Not enough data to print examples.")
    print("="*60 + "\n")

    # 6. Truncate any trailing partial example if earlier skips broke alignment.
    def truncate_to_block(data_list, block_size, name):
        remainder = len(data_list) % block_size
        if remainder != 0:
            print(f"Truncating {name} data by {remainder} tokens to align with block size.")
            return data_list[:-remainder]
        return data_list

    train_data = truncate_to_block(train_data, data_size, "train")
    val_data = truncate_to_block(val_data, data_size, "val")

    # 7. Convert to uint16 arrays and sanity-check the vocabulary range.
    train_bin = np.array(train_data, dtype=np.uint16)
    val_bin = np.array(val_data, dtype=np.uint16)

    assert train_bin.max() < vocab_size, f"Dirty data detected! Max token {train_bin.max()} >= vocab_size {vocab_size}"
    if len(val_bin) > 0:
        assert val_bin.max() < vocab_size, f"Dirty data detected! Max token {val_bin.max()} >= vocab_size {vocab_size}"

    # Save flattened binary files.
    train_bin.tofile(os.path.join(args.out_dir, 'train.bin'))
    val_bin.tofile(os.path.join(args.out_dir, 'val.bin'))

    # 8. Save metadata describing the serialization protocol.
    meta = {
        'format_version': 'fixed_length_char_v1',
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
        'block_size': data_size - 1,
        'quiz_size': quiz_size,
        'response_size': response_size,
        'data_size': data_size,
        'max_quiz_len': max_quiz_len,
        'max_response_len': max_response_len,
        'max_input_len': max_quiz_len,
        'max_output_len': max_response_len,
        'input_key': args.input_key,
        'output_key': args.output_key,
        'special_tokens': special_tokens,
        'pad_token': '<PAD>',
        'sep_token': '<SEP>',
        'eos_token': '<EOS>',
        'mask_token': '<MASK>',
        'dollar_token': '$',
        'tokenizer_type': 'char',
        'serialization': 'quiz_pad_sep + response_pad_eos',
        'dtype': 'uint16',
        'data_path': args.data_path,
        'val_ratio': args.val_ratio,
        'seed': args.seed,
    }

    meta_path = os.path.join(args.out_dir, args.meta_name)
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    print(f"✅ Data successfully prepared in '{args.out_dir}'.")
    print(f"   Saved train.bin, val.bin, and {args.meta_name}.")

if __name__ == "__main__":
    main()
