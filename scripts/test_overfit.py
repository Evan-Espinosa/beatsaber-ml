"""
Overfitting Test for Beat Saber Generator

This script trains the model on a single sample to verify:
1. The model architecture is correct
2. Gradients flow properly
3. The model can learn (loss decreases)

If the loss doesn't decrease, something is broken.

Usage:
    python scripts/test_overfit.py
    python scripts/test_overfit.py --steps 500 --model base
"""

import argparse
import torch
import torch.nn.functional as F

from src.data.dataset import BeatSaberDataset
from src.models.generator import (
    BeatSaberGenerator,
    BeatSaberGeneratorSmall,
    BeatSaberGeneratorLarge
)


def run_overfit_test(
    data_dir: str = "data/processed",
    max_audio_len: int = 500,
    max_token_len: int = 100,
    steps: int = 200,
    lr: float = 5e-4,
    model_size: str = "small",
    sample_idx: int = 0
):
    """
    Run overfitting test on a single sample.

    Args:
        data_dir: Path to processed data
        max_audio_len: Maximum audio sequence length
        max_token_len: Maximum token sequence length
        steps: Number of training steps
        lr: Learning rate
        model_size: Model size ('small', 'base', 'large')
        sample_idx: Which sample to overfit on
    """
    print("=" * 50)
    print("OVERFITTING TEST")
    print("=" * 50)

    # Load dataset
    print(f"\nLoading dataset from {data_dir}...")
    dataset = BeatSaberDataset(
        data_dir,
        max_audio_len=max_audio_len,
        max_token_len=max_token_len
    )

    # Get single sample
    sample = dataset[sample_idx]
    print(f"Sample: {sample['metadata']['song_name']}")
    print(f"  Audio shape: {sample['audio_features'].shape}")
    print(f"  Token shape: {sample['token_ids'].shape}")
    print(f"  Difficulty: {sample['difficulty']}")

    # Create model
    print(f"\nCreating {model_size} model...")
    if model_size == "small":
        model = BeatSaberGeneratorSmall()
    elif model_size == "large":
        model = BeatSaberGeneratorLarge()
    else:
        model = BeatSaberGenerator()

    print(f"  Parameters: {model.count_parameters() / 1e6:.1f}M")

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepare data (batch size 1)
    audio = sample['audio_features'].unsqueeze(0)
    tokens = sample['token_ids'].unsqueeze(0)
    difficulty = torch.tensor([sample['difficulty']])

    # Teacher forcing setup
    input_ids = tokens[:, :-1]   # Input: all tokens except last
    target_ids = tokens[:, 1:]   # Target: all tokens except first

    print(f"\nTraining for {steps} steps...")
    print("-" * 40)

    # Training loop
    model.train()
    initial_loss = None
    final_loss = None

    for step in range(steps):
        # Forward pass
        logits = model(audio, input_ids, difficulty)

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=0  # Ignore PAD tokens
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track loss
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

        # Print progress
        if step % (steps // 8) == 0 or step == steps - 1:
            print(f"Step {step:4d}: loss = {loss.item():.4f}")

    # Results
    print("-" * 40)
    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Reduction:    {(1 - final_loss/initial_loss) * 100:.1f}%")

    print("\n" + "=" * 50)
    if final_loss < 0.5:
        print("RESULT: PASSED - Model can learn!")
    elif final_loss < 2.0:
        print("RESULT: OK - Loss is decreasing, try more steps")
    else:
        print("RESULT: FAILED - Loss not decreasing, check model/data")
    print("=" * 50)

    return final_loss < 2.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overfitting test for Beat Saber generator")
    parser.add_argument("--data", default="data/processed", help="Data directory")
    parser.add_argument("--steps", type=int, default=200, help="Training steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--model", choices=["small", "base", "large"], default="small")
    parser.add_argument("--max-audio", type=int, default=500, help="Max audio length")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max token length")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to use")

    args = parser.parse_args()

    success = run_overfit_test(
        data_dir=args.data,
        max_audio_len=args.max_audio,
        max_token_len=args.max_tokens,
        steps=args.steps,
        lr=args.lr,
        model_size=args.model,
        sample_idx=args.sample
    )

    exit(0 if success else 1)
