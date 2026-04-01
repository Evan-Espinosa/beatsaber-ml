"""
Preprocessing Pipeline for Beat Saber ML

Converts raw map folders into training-ready .pt files.

Pipeline:
1. Parse map folder (Info.dat + difficulty files)
2. Filter by criteria (BPM changes, minimum notes, etc.)
3. Extract audio features aligned to tick grid
4. Tokenize events
5. Save as .pt file

Output format:
{
    "audio_features": Tensor[n_ticks, feature_dim],
    "token_ids": Tensor[n_tokens],
    "metadata": {
        "map_id": str,
        "song_name": str,
        "difficulty": str,
        "bpm": float,
        "duration": float,
        "n_notes": int,
        ...
    }
}
"""

import json
import logging
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .parser import BeatSaberParser, CanonicalMap
from .features import AudioFeatureExtractor, AudioFeatures
from .tokenizer import EventTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline."""
    # Feature extraction
    n_mels: int = 128
    context_frames: int = 5
    sample_rate: int = 22050
    ticks_per_beat: int = 16

    # Filtering
    min_notes: int = 50           # Minimum notes to include
    max_duration: float = 600.0   # Maximum duration in seconds
    skip_bpm_changes: bool = True # Skip maps with BPM changes
    difficulties: Optional[list] = None  # Filter by difficulty names

    # Output
    use_float16: bool = False     # Save features as float16 to save space
    compress: bool = False        # Compress output files


class PreprocessingPipeline:
    """
    Full preprocessing pipeline for Beat Saber maps.

    Usage:
        pipeline = PreprocessingPipeline(config)
        result = pipeline.process_map(Path("data/raw/abc123"))

        # Or batch process
        pipeline.process_all(
            input_dir=Path("data/raw"),
            output_dir=Path("data/processed")
        )
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

        # Initialize components
        self.parser = BeatSaberParser(ticks_per_beat=self.config.ticks_per_beat)
        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=self.config.sample_rate,
            n_mels=self.config.n_mels,
            context_frames=self.config.context_frames
        )
        self.tokenizer = EventTokenizer()

    def process_map(
        self,
        map_folder: Path,
        output_dir: Optional[Path] = None
    ) -> list[dict]:
        """
        Process a single map folder.

        Args:
            map_folder: Path to extracted map folder
            output_dir: Optional output directory for .pt files

        Returns:
            List of processed sample dicts (one per difficulty)
        """
        map_folder = Path(map_folder)
        results = []

        try:
            # Parse map
            canonical_maps = self.parser.parse_map_folder(map_folder)
        except Exception as e:
            logger.warning(f"Failed to parse {map_folder}: {e}")
            return []

        # Find audio file
        audio_path = None
        for ext in ['.egg', '.ogg', '.mp3', '.wav']:
            candidates = list(map_folder.glob(f"*{ext}"))
            if candidates:
                audio_path = candidates[0]
                break

        if not audio_path:
            logger.warning(f"No audio file found in {map_folder}")
            return []

        # Process each difficulty
        for canonical_map in canonical_maps:
            # Apply filters
            if not self._should_process(canonical_map):
                continue

            try:
                sample = self._process_difficulty(
                    canonical_map,
                    audio_path,
                    map_folder.name
                )

                if sample:
                    results.append(sample)

                    # Save if output directory specified
                    if output_dir:
                        self._save_sample(sample, output_dir)

            except Exception as e:
                logger.warning(
                    f"Failed to process {map_folder.name}/{canonical_map.difficulty.name}: {e}"
                )
                continue

        return results

    def _should_process(self, canonical_map: CanonicalMap) -> bool:
        """Check if map passes filtering criteria."""
        # Skip if BPM changes present (Phase 1)
        if self.config.skip_bpm_changes and canonical_map.metadata.has_bpm_changes:
            logger.debug(f"Skipping {canonical_map.difficulty.name}: has BPM changes")
            return False

        # Check minimum notes
        if canonical_map.note_count < self.config.min_notes:
            logger.debug(f"Skipping {canonical_map.difficulty.name}: only {canonical_map.note_count} notes")
            return False

        # Check duration
        if canonical_map.duration_seconds > self.config.max_duration:
            logger.debug(f"Skipping {canonical_map.difficulty.name}: too long ({canonical_map.duration_seconds}s)")
            return False

        # Check difficulty filter
        if self.config.difficulties:
            if canonical_map.difficulty.name not in self.config.difficulties:
                return False

        return True

    def _process_difficulty(
        self,
        canonical_map: CanonicalMap,
        audio_path: Path,
        map_id: str
    ) -> Optional[dict]:
        """Process a single difficulty."""
        # Extract audio features
        audio_features = self.feature_extractor.extract(
            audio_path,
            bpm=canonical_map.metadata.bpm,
            ticks_per_beat=self.config.ticks_per_beat
        )

        # Tokenize events
        token_ids = self.tokenizer.encode(canonical_map.events)

        # Convert to tensors
        features_tensor = torch.from_numpy(audio_features.features)
        tokens_tensor = torch.tensor(token_ids, dtype=torch.long)

        # Use float16 if configured
        if self.config.use_float16:
            features_tensor = features_tensor.half()

        # Build metadata
        metadata = {
            "map_id": map_id,
            "song_name": canonical_map.metadata.song_name,
            "song_author": canonical_map.metadata.song_author,
            "level_author": canonical_map.metadata.level_author,
            "difficulty": canonical_map.difficulty.name,
            "characteristic": canonical_map.difficulty.characteristic,
            "bpm": canonical_map.metadata.bpm,
            "duration": audio_features.duration,
            "n_ticks": audio_features.n_ticks,
            "n_tokens": len(token_ids),
            "n_notes": canonical_map.note_count,
            "n_bombs": canonical_map.bomb_count,
            "n_obstacles": canonical_map.obstacle_count,
            "format_version": canonical_map.format_version,
            "njs": canonical_map.difficulty.njs,
        }

        return {
            "audio_features": features_tensor,
            "token_ids": tokens_tensor,
            "metadata": metadata
        }

    def _save_sample(self, sample: dict, output_dir: Path):
        """Save a processed sample to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename
        map_id = sample["metadata"]["map_id"]
        difficulty = sample["metadata"]["difficulty"]
        characteristic = sample["metadata"]["characteristic"]
        filename = f"{map_id}_{characteristic}_{difficulty}.pt"
        output_path = output_dir / filename

        # Save
        torch.save(sample, output_path)
        logger.debug(f"Saved: {output_path}")

    def process_all(
        self,
        input_dir: Path,
        output_dir: Path,
        max_workers: int = 1,  # Set to 1 for now due to librosa threading issues
        limit: Optional[int] = None
    ) -> dict:
        """
        Process all maps in a directory.

        Args:
            input_dir: Directory containing map folders
            output_dir: Output directory for .pt files
            max_workers: Number of parallel workers
            limit: Maximum number of maps to process

        Returns:
            Summary dict with processing statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find map folders
        map_folders = [
            d for d in input_dir.iterdir()
            if d.is_dir() and (d / "Info.dat").exists() or (d / "info.dat").exists()
        ]

        if limit:
            map_folders = map_folders[:limit]

        logger.info(f"Processing {len(map_folders)} maps...")

        stats = {
            "total_maps": len(map_folders),
            "processed_maps": 0,
            "processed_difficulties": 0,
            "failed_maps": 0,
            "skipped_difficulties": 0
        }

        # Process maps
        for map_folder in tqdm(map_folders, desc="Processing maps"):
            try:
                results = self.process_map(map_folder, output_dir)
                if results:
                    stats["processed_maps"] += 1
                    stats["processed_difficulties"] += len(results)
                else:
                    stats["failed_maps"] += 1
            except Exception as e:
                logger.error(f"Error processing {map_folder}: {e}")
                stats["failed_maps"] += 1

        # Save summary
        summary_path = output_dir / "preprocessing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Processing complete: {stats['processed_difficulties']} difficulties from {stats['processed_maps']} maps")
        logger.info(f"Failed: {stats['failed_maps']} maps")

        return stats


def preprocess_dataset(
    input_dir: str = "data/raw",
    output_dir: str = "data/processed",
    **config_kwargs
) -> dict:
    """
    Convenience function to preprocess a dataset.

    Args:
        input_dir: Directory containing raw map folders
        output_dir: Output directory for processed files
        **config_kwargs: Additional PreprocessConfig arguments

    Returns:
        Processing statistics
    """
    config = PreprocessConfig(**config_kwargs)
    pipeline = PreprocessingPipeline(config)

    return pipeline.process_all(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir)
    )


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess Beat Saber maps')
    parser.add_argument('--input', '-i', default='data/raw', help='Input directory')
    parser.add_argument('--output', '-o', default='data/processed', help='Output directory')
    parser.add_argument('--limit', type=int, help='Maximum maps to process')
    parser.add_argument('--min-notes', type=int, default=50, help='Minimum notes per difficulty')
    parser.add_argument('--float16', action='store_true', help='Use float16 for features')
    parser.add_argument('--include-bpm-changes', action='store_true', help='Include maps with BPM changes')
    parser.add_argument('--difficulties', nargs='+', help='Only process specific difficulties')
    parser.add_argument('--single', help='Process single map folder')

    args = parser.parse_args()

    config = PreprocessConfig(
        min_notes=args.min_notes,
        use_float16=args.float16,
        skip_bpm_changes=not args.include_bpm_changes,
        difficulties=args.difficulties
    )

    pipeline = PreprocessingPipeline(config)

    if args.single:
        # Process single map
        results = pipeline.process_map(Path(args.single), Path(args.output))
        print(f"\nProcessed {len(results)} difficulties")
        for r in results:
            m = r["metadata"]
            print(f"  {m['difficulty']}: {m['n_notes']} notes, {m['n_tokens']} tokens")
    else:
        # Process all maps
        stats = pipeline.process_all(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            limit=args.limit
        )
        print(f"\n=== Summary ===")
        print(f"Total maps: {stats['total_maps']}")
        print(f"Processed: {stats['processed_maps']} maps, {stats['processed_difficulties']} difficulties")
        print(f"Failed: {stats['failed_maps']} maps")
