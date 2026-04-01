"""
PyTorch Dataset for Beat Saber ML

Loads preprocessed .pt files for training the generator model.

Each sample contains:
- audio_features: [n_ticks, feature_dim]
- token_ids: [n_tokens]
- metadata: dict with map info
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Difficulty name to integer mapping
DIFFICULTY_MAP = {
    'Easy': 0,
    'Normal': 1,
    'Hard': 2,
    'Expert': 3,
    'ExpertPlus': 4,
    'Expert+': 4
}


class BeatSaberDataset(Dataset):
    """
    Dataset for loading preprocessed Beat Saber maps.

    Each sample returns:
    - audio_features: [T_audio, D_audio] tensor
    - token_ids: [T_tokens] tensor
    - difficulty: int (0-4)
    - metadata: dict

    Example:
        dataset = BeatSaberDataset('data/processed')
        sample = dataset[0]
        print(f"Audio shape: {sample['audio_features'].shape}")
        print(f"Tokens: {sample['token_ids'].shape}")
    """

    def __init__(
        self,
        data_dir: str,
        split: Optional[str] = None,
        max_audio_len: Optional[int] = None,
        max_token_len: Optional[int] = None,
        difficulties: Optional[list] = None,
        cache_in_memory: bool = False
    ):
        """
        Args:
            data_dir: Directory containing .pt files
            split: Optional split ('train', 'val', 'test') - looks for subdir
            max_audio_len: Maximum audio sequence length (truncate if longer)
            max_token_len: Maximum token sequence length (truncate if longer)
            difficulties: List of difficulty names to include (None = all)
            cache_in_memory: If True, load all samples into memory
        """
        self.data_dir = Path(data_dir)
        self.max_audio_len = max_audio_len
        self.max_token_len = max_token_len
        self.difficulties = difficulties
        self.cache_in_memory = cache_in_memory

        # Handle split subdirectory
        if split:
            split_dir = self.data_dir / split
            if split_dir.exists():
                self.data_dir = split_dir

        # Find all .pt files
        self.sample_paths = sorted(self.data_dir.glob('*.pt'))

        if not self.sample_paths:
            raise ValueError(f"No .pt files found in {self.data_dir}")

        # Filter by difficulty if specified
        if self.difficulties:
            self.sample_paths = [
                p for p in self.sample_paths
                if any(d in p.stem for d in self.difficulties)
            ]

        logger.info(f"Found {len(self.sample_paths)} samples in {self.data_dir}")

        # Cache samples if requested
        self._cache = {}
        if self.cache_in_memory:
            logger.info("Loading samples into memory...")
            for i, path in enumerate(self.sample_paths):
                self._cache[i] = torch.load(path)
            logger.info(f"Cached {len(self._cache)} samples")

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> dict:
        # Load from cache or disk
        if idx in self._cache:
            sample = self._cache[idx]
        else:
            sample = torch.load(self.sample_paths[idx])

        audio_features = sample['audio_features']
        token_ids = sample['token_ids']
        metadata = sample['metadata']

        # Convert to float32 if stored as float16
        if audio_features.dtype == torch.float16:
            audio_features = audio_features.float()

        # Truncate if necessary
        if self.max_audio_len and audio_features.size(0) > self.max_audio_len:
            audio_features = audio_features[:self.max_audio_len]

        if self.max_token_len and token_ids.size(0) > self.max_token_len:
            token_ids = token_ids[:self.max_token_len]

        # Get difficulty as integer
        difficulty_name = metadata.get('difficulty', 'Expert')
        difficulty = DIFFICULTY_MAP.get(difficulty_name, 3)

        return {
            'audio_features': audio_features,
            'token_ids': token_ids,
            'difficulty': difficulty,
            'metadata': metadata
        }

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        audio_lens = []
        token_lens = []
        difficulties = []

        for i in range(min(len(self), 100)):  # Sample first 100
            sample = self[i]
            audio_lens.append(sample['audio_features'].size(0))
            token_lens.append(sample['token_ids'].size(0))
            difficulties.append(sample['difficulty'])

        return {
            'num_samples': len(self),
            'avg_audio_len': sum(audio_lens) / len(audio_lens),
            'max_audio_len': max(audio_lens),
            'avg_token_len': sum(token_lens) / len(token_lens),
            'max_token_len': max(token_lens),
            'feature_dim': self[0]['audio_features'].size(1)
        }


def collate_fn(batch: list, pad_id: int = 0) -> dict:
    """
    Collate function for batching variable-length sequences.

    Pads audio features and token sequences to the same length within batch.

    Args:
        batch: List of sample dicts from dataset
        pad_id: Token ID to use for padding (default 0 = PAD)

    Returns:
        Batched tensors with padding
    """
    audio_features = [item['audio_features'] for item in batch]
    token_ids = [item['token_ids'] for item in batch]
    difficulties = torch.tensor([item['difficulty'] for item in batch])

    # Get max lengths in this batch
    max_audio_len = max(f.size(0) for f in audio_features)
    max_token_len = max(t.size(0) for t in token_ids)

    # Get feature dimension
    feature_dim = audio_features[0].size(1)

    # Pad audio features
    audio_padded = torch.zeros(len(batch), max_audio_len, feature_dim)
    audio_mask = torch.zeros(len(batch), max_audio_len, dtype=torch.bool)

    for i, feat in enumerate(audio_features):
        audio_len = feat.size(0)
        audio_padded[i, :audio_len] = feat
        audio_mask[i, :audio_len] = True

    # Pad token sequences
    tokens_padded = torch.full((len(batch), max_token_len), pad_id, dtype=torch.long)
    tokens_mask = torch.zeros(len(batch), max_token_len, dtype=torch.bool)

    for i, tokens in enumerate(token_ids):
        token_len = tokens.size(0)
        tokens_padded[i, :token_len] = tokens
        tokens_mask[i, :token_len] = True

    return {
        'audio_features': audio_padded,      # [B, T_audio, D]
        'audio_mask': audio_mask,            # [B, T_audio]
        'token_ids': tokens_padded,          # [B, T_tokens]
        'token_mask': tokens_mask,           # [B, T_tokens]
        'difficulty': difficulties,          # [B]
    }


def create_data_loaders(
    data_dir: str,
    batch_size: int = 8,
    val_split: float = 0.1,
    num_workers: int = 0,
    **dataset_kwargs
) -> tuple:
    """
    Create train and validation data loaders.

    If data_dir has train/val subdirectories, uses those.
    Otherwise, randomly splits the data.

    Args:
        data_dir: Directory containing .pt files
        batch_size: Batch size
        val_split: Fraction of data for validation (if no split dirs)
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional args for BeatSaberDataset

    Returns:
        (train_loader, val_loader)
    """
    data_dir = Path(data_dir)

    # Check for existing splits
    if (data_dir / 'train').exists() and (data_dir / 'val').exists():
        train_dataset = BeatSaberDataset(data_dir, split='train', **dataset_kwargs)
        val_dataset = BeatSaberDataset(data_dir, split='val', **dataset_kwargs)
    else:
        # Create random split
        full_dataset = BeatSaberDataset(data_dir, **dataset_kwargs)

        n_val = int(len(full_dataset) * val_split)
        n_train = len(full_dataset) - n_val

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Split dataset: {n_train} train, {n_val} val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test Beat Saber dataset')
    parser.add_argument('--data', default='data/processed', help='Data directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')

    args = parser.parse_args()

    # Test dataset
    dataset = BeatSaberDataset(args.data)
    print(f"\n=== Dataset ===")
    print(f"Samples: {len(dataset)}")

    stats = dataset.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Test single sample
    sample = dataset[0]
    print(f"\n=== Sample 0 ===")
    print(f"Audio features: {sample['audio_features'].shape}")
    print(f"Token IDs: {sample['token_ids'].shape}")
    print(f"Difficulty: {sample['difficulty']}")
    print(f"Song: {sample['metadata']['song_name']}")

    # Test data loader
    train_loader, val_loader = create_data_loaders(
        args.data,
        batch_size=args.batch_size
    )

    print(f"\n=== DataLoader ===")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test batch
    batch = next(iter(train_loader))
    print(f"\n=== Batch ===")
    print(f"Audio features: {batch['audio_features'].shape}")
    print(f"Audio mask: {batch['audio_mask'].shape}")
    print(f"Token IDs: {batch['token_ids'].shape}")
    print(f"Token mask: {batch['token_mask'].shape}")
    print(f"Difficulty: {batch['difficulty']}")
