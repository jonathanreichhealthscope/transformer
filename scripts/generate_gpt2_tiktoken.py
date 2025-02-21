#!/usr/bin/env python3
import json
import tiktoken
import os

def generate_gpt2_files():
    # Get the GPT-2 encoding
    enc = tiktoken.get_encoding("gpt2")
    
    # Create tiktoken_data directory if it doesn't exist
    os.makedirs("scripts/tiktoken_data", exist_ok=True)
    
    # Generate vocabulary file
    vocab = {}
    # Get all possible tokens and their byte values
    for i in range(enc.n_vocab):
        token_bytes = enc.decode_single_token_bytes(i)
        vocab[token_bytes.decode('utf-8', errors='replace')] = i
    
    with open("scripts/tiktoken_data/gpt2.vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # Get the merges from the encoder's BPE ranks
    merges = enc._mergeable_ranks
    merge_list = []
    
    # Debug print to see the structure
    print("First merge pair:", next(iter(merges.keys())))
    
    for merge_str in sorted(merges.keys(), key=lambda x: merges[x]):
        if isinstance(merge_str, bytes):
            merge_str = merge_str.decode('utf-8', errors='replace')
        merge_list.append(merge_str)
    
    with open("scripts/tiktoken_data/gpt2.merges.json", "w", encoding="utf-8") as f:
        json.dump(merge_list, f, ensure_ascii=False, indent=2)
    
    print("Generated files:")
    print("- scripts/tiktoken_data/gpt2.vocab.json")
    print("- scripts/tiktoken_data/gpt2.merges.json")

if __name__ == "__main__":
    generate_gpt2_files() 