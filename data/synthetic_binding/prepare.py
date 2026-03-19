"""
Prepare a synthetic "binding problem" dataset for evaluating cross-head mixing.

Vocabulary (50 tokens):
  0-9   : Colors
  10-19 : Shapes
  20-49 : Targets

Each document is a flat sequence of [Color, Shape, Target] triplets.
Within a document, a fixed mapping (color, shape) -> target is established,
with deliberate collisions: the same color appears with multiple shapes
(and vice versa), each mapping to a DIFFERENT target. This forces the model
to bind both attributes to predict the target correctly.

Outputs train.bin, val.bin (uint16), and meta.pkl.
"""

import os
import pickle
import numpy as np

# --- Config ---
N_COLORS = 10       # token IDs 0..9
N_SHAPES = 10       # token IDs 10..19
N_TARGETS = 30      # token IDs 20..49
VOCAB_SIZE = N_COLORS + N_SHAPES + N_TARGETS  # 50

TRIPLETS_PER_DOC = 128   # 384 tokens per document
N_DOCS_TRAIN = 5000
N_DOCS_VAL = 500
GRID_SIZE = 3  # 3x3 collision grid per document

SEED = 42
# --------------

def generate_document(rng):
    """Generate one document: 128 triplets from a random binding rule set.

    Strategy: pick 3 colors and 3 shapes to form a 3x3 grid (9 pairs).
    Each pair gets a unique target. This guarantees every color appears with
    3 different shapes and every shape with 3 different colors — the model
    MUST bind both to predict the target.
    """
    # Pick grid colors and shapes (without replacement)
    grid_colors = rng.choice(N_COLORS, size=GRID_SIZE, replace=False)
    grid_shapes = rng.choice(N_SHAPES, size=GRID_SIZE, replace=False)

    # Assign a unique target to each (color, shape) pair in the grid
    targets = rng.choice(N_TARGETS, size=GRID_SIZE * GRID_SIZE, replace=False)
    rules = {}
    idx = 0
    for c in grid_colors:
        for s in grid_shapes:
            rules[(int(c), int(s))] = int(targets[idx]) + N_COLORS + N_SHAPES  # offset to target IDs
            idx += 1

    # Generate triplets by sampling uniformly from the rule set
    rule_keys = list(rules.keys())
    tokens = []
    for _ in range(TRIPLETS_PER_DOC):
        c, s = rule_keys[rng.integers(len(rule_keys))]
        t = rules[(c, s)]
        tokens.extend([c, s + N_COLORS, t])  # offset shape to its ID range

    return np.array(tokens, dtype=np.uint16)


def main():
    rng = np.random.default_rng(SEED)
    out_dir = os.path.dirname(__file__)

    # Generate documents
    print(f"Generating {N_DOCS_TRAIN} train + {N_DOCS_VAL} val documents...")
    train_docs = [generate_document(rng) for _ in range(N_DOCS_TRAIN)]
    val_docs = [generate_document(rng) for _ in range(N_DOCS_VAL)]

    train_ids = np.concatenate(train_docs)
    val_ids = np.concatenate(val_docs)
    print(f"train: {len(train_ids):,} tokens")
    print(f"val:   {len(val_ids):,} tokens")

    # Save binary files
    train_ids.tofile(os.path.join(out_dir, 'train.bin'))
    val_ids.tofile(os.path.join(out_dir, 'val.bin'))

    # Build itos/stoi for compatibility with nanoGPT's meta.pkl format
    color_names = [f'C{i}' for i in range(N_COLORS)]
    shape_names = [f'S{i}' for i in range(N_SHAPES)]
    target_names = [f'T{i}' for i in range(N_TARGETS)]
    all_names = color_names + shape_names + target_names

    itos = {i: name for i, name in enumerate(all_names)}
    stoi = {name: i for i, name in enumerate(all_names)}

    meta = {
        'vocab_size': VOCAB_SIZE,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"Saved to {out_dir}/")
    print(f"vocab_size = {VOCAB_SIZE}")
    print(f"Example triplet: {itos[train_ids[0]]} {itos[train_ids[1]]} {itos[train_ids[2]]}")


if __name__ == '__main__':
    main()
