import json
from collections import Counter
import re

def tokenize(text):
    """Simple tokenization by splitting on spaces and keeping punctuation."""
    return re.findall(r'\b\w+\b|[^\w\s]', text.lower())

def build_vocab(training_file, min_freq=2):
    # Read training data
    with open(training_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Collect all words
    word_counter = Counter()
    for line in lines:
        if '|' in line:
            input_text, output_text = line.strip().split('|')
            words = tokenize(input_text) + tokenize(output_text)
            word_counter.update(words)

    # Build vocabulary with words that appear at least min_freq times
    vocab = {}
    special_tokens = {
        "<|endoftext|>": 50256,
        "<|pad|>": 50257,
        "<|bos|>": 50258,
        "<|eos|>": 50259,
        "<|unk|>": 50260
    }
    
    # Add special tokens first
    vocab.update(special_tokens)
    
    # Add frequent words
    current_id = 0
    for word, count in word_counter.most_common():
        if count >= min_freq and word not in vocab:
            vocab[word] = current_id
            current_id += 1

    print(f"Vocabulary size (including special tokens): {len(vocab)}")
    
    # Save vocabulary
    with open('custom.vocab', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)
    
    # Create a simple merge rules file (byte-level for simplicity)
    merges = []
    with open('custom.merges', 'w', encoding='utf-8') as f:
        for merge in merges:
            f.write(f"{merge}\n")

    print("Created custom.vocab and custom.merges")
    return vocab

if __name__ == "__main__":
    vocab = build_vocab("../../data/training_pairs.txt")
    print("Most common tokens:", list(vocab.items())[:20]) 