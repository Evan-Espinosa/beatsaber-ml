"""
Beat Saber Map Generation Pipeline

Full pipeline for generating playable Beat Saber maps:
1. Load trained model
2. Extract audio features
3. Generate token sequence
4. Decode to canonical events
5. Validate and fix constraints
6. Export to .dat files

Usage:
    python -m src.generation.generate --audio song.ogg --output output_map/
    python -m src.generation.generate --audio song.ogg --difficulty Expert --checkpoint best_model.pt
"""

import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch

from src.data.features import AudioFeatureExtractor
from src.data.tokenizer import EventTokenizer
from src.models.generator import (
    BeatSaberGenerator,
    BeatSaberGeneratorSmall,
    BeatSaberGeneratorLarge
)
from src.evaluation.constraints import ConstraintValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Difficulty mapping
DIFFICULTY_MAP = {
    "Easy": 0,
    "Normal": 1,
    "Hard": 2,
    "Expert": 3,
    "ExpertPlus": 4,
    "Expert+": 4
}

DIFFICULTY_RANK = {
    "Easy": 1,
    "Normal": 3,
    "Hard": 5,
    "Expert": 7,
    "ExpertPlus": 9
}


class MapGenerator:
    """
    Full pipeline for generating Beat Saber maps from audio.

    Example:
        generator = MapGenerator.from_checkpoint("checkpoints/best_model.pt")
        generator.generate(
            audio_path="song.ogg",
            output_dir="output_map/",
            difficulty="Expert",
            bpm=120
        )
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: EventTokenizer,
        feature_extractor: AudioFeatureExtractor,
        validator: ConstraintValidator,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.validator = validator
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_size: str = "small",
        device: str = "cuda"
    ) -> "MapGenerator":
        """
        Load generator from a trained checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            model_size: Model size ("small", "base", "large")
            device: Device to use

        Returns:
            MapGenerator instance
        """
        # Create components
        tokenizer = EventTokenizer()
        feature_extractor = AudioFeatureExtractor()
        validator = ConstraintValidator()

        # Create model
        if model_size == "small":
            model = BeatSaberGeneratorSmall(
                vocab_size=tokenizer.vocab_size,
                d_audio=feature_extractor.feature_dim
            )
        elif model_size == "large":
            model = BeatSaberGeneratorLarge(
                vocab_size=tokenizer.vocab_size,
                d_audio=feature_extractor.feature_dim
            )
        else:
            model = BeatSaberGenerator(
                vocab_size=tokenizer.vocab_size,
                d_audio=feature_extractor.feature_dim
            )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return cls(model, tokenizer, feature_extractor, validator, device)

    def generate(
        self,
        audio_path: str,
        output_dir: str,
        difficulty: str = "Expert",
        bpm: float = 120.0,
        song_name: Optional[str] = None,
        song_author: Optional[str] = None,
        level_author: str = "BeatSaber-ML",
        max_length: int = 4000,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: Optional[float] = None,
        validate: bool = True,
        n_candidates: int = 1
    ) -> Path:
        """
        Generate a Beat Saber map from an audio file.

        Args:
            audio_path: Path to audio file (.ogg, .egg, .mp3, .wav)
            output_dir: Output directory for map files
            difficulty: Difficulty level
            bpm: Beats per minute
            song_name: Song name (defaults to filename)
            song_author: Song author
            level_author: Level author
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            validate: Whether to validate and fix constraints
            n_candidates: Number of candidates to generate (best is selected)

        Returns:
            Path to output directory
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Defaults
        if song_name is None:
            song_name = audio_path.stem

        logger.info(f"Generating {difficulty} map for: {song_name}")

        # Extract audio features
        logger.info("Extracting audio features...")
        audio_features = self.feature_extractor.extract(
            audio_path,
            bpm=bpm,
            ticks_per_beat=16
        )
        logger.info(f"  Duration: {audio_features.duration:.1f}s")
        logger.info(f"  Ticks: {audio_features.n_ticks}")

        # Prepare input
        features_tensor = torch.from_numpy(audio_features.features).to(self.device)
        difficulty_id = DIFFICULTY_MAP.get(difficulty, 3)
        difficulty_tensor = torch.tensor([difficulty_id]).to(self.device)

        # Generate candidates
        best_events = None
        best_score = float("-inf")

        for i in range(n_candidates):
            logger.info(f"Generating candidate {i+1}/{n_candidates}...")

            # Generate tokens
            with torch.no_grad():
                generated = self.model.generate(
                    features_tensor.unsqueeze(0),
                    difficulty_tensor,
                    max_len=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )

            # Decode to events
            token_ids = generated[0].cpu().tolist()
            events = self.tokenizer.decode(token_ids)

            # Add timing info
            events = self._add_timing(events, bpm)

            logger.info(f"  Generated {len(events)} events")

            # Validate and fix
            if validate:
                is_valid, violations = self.validator.validate(events)
                if not is_valid:
                    logger.info(f"  Found {len(violations)} violations, fixing...")
                    events = self.validator.fix(events)

            # Score (simple: more notes = better, within reason)
            stats = self.validator.get_stats(events)
            score = stats.get("notes", 0)

            if score > best_score:
                best_score = score
                best_events = events

        events = best_events
        logger.info(f"Selected best candidate with {len(events)} events")

        # Export to .dat files
        output_dir.mkdir(parents=True, exist_ok=True)

        self._export_info_dat(
            output_dir,
            song_name=song_name,
            song_author=song_author or "Unknown",
            level_author=level_author,
            bpm=bpm,
            difficulty=difficulty,
            audio_filename=audio_path.name
        )

        self._export_difficulty_dat(
            output_dir,
            events=events,
            difficulty=difficulty,
            bpm=bpm
        )

        # Copy audio file
        audio_dest = output_dir / "song.egg"
        shutil.copy(audio_path, audio_dest)
        logger.info(f"Copied audio to {audio_dest}")

        logger.info(f"Map generated successfully: {output_dir}")
        return output_dir

    def _add_timing(self, events: list, bpm: float) -> list:
        """Add t_sec timing to events based on tick and BPM."""
        ticks_per_beat = 16
        seconds_per_tick = 60.0 / (bpm * ticks_per_beat)

        for event in events:
            event["t_sec"] = event["tick"] * seconds_per_tick

        return events

    def _export_info_dat(
        self,
        output_dir: Path,
        song_name: str,
        song_author: str,
        level_author: str,
        bpm: float,
        difficulty: str,
        audio_filename: str
    ):
        """Export Info.dat file."""
        # Normalize difficulty name
        if difficulty == "Expert+":
            difficulty = "ExpertPlus"

        info = {
            "_version": "2.0.0",
            "_songName": song_name,
            "_songSubName": "",
            "_songAuthorName": song_author,
            "_levelAuthorName": level_author,
            "_beatsPerMinute": bpm,
            "_songTimeOffset": 0,
            "_shuffle": 0,
            "_shufflePeriod": 0.5,
            "_previewStartTime": 12,
            "_previewDuration": 10,
            "_songFilename": "song.egg",
            "_coverImageFilename": "cover.jpg",
            "_environmentName": "DefaultEnvironment",
            "_allDirectionsEnvironmentName": "GlassDesertEnvironment",
            "_difficultyBeatmapSets": [
                {
                    "_beatmapCharacteristicName": "Standard",
                    "_difficultyBeatmaps": [
                        {
                            "_difficulty": difficulty,
                            "_difficultyRank": DIFFICULTY_RANK.get(difficulty, 7),
                            "_beatmapFilename": f"{difficulty}Standard.dat",
                            "_noteJumpMovementSpeed": 16,
                            "_noteJumpStartBeatOffset": 0
                        }
                    ]
                }
            ]
        }

        info_path = output_dir / "Info.dat"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        logger.info(f"Wrote {info_path}")

    def _export_difficulty_dat(
        self,
        output_dir: Path,
        events: list,
        difficulty: str,
        bpm: float
    ):
        """Export difficulty .dat file in v2 format."""
        # Normalize difficulty name
        if difficulty == "Expert+":
            difficulty = "ExpertPlus"

        ticks_per_beat = 16

        # Convert events to v2 format
        notes = []
        obstacles = []

        for event in events:
            tick = event["tick"]
            beat_time = tick / ticks_per_beat
            data = event["data"]

            if event["type"] == "note":
                notes.append({
                    "_time": beat_time,
                    "_lineIndex": data["lane"],
                    "_lineLayer": data["row"],
                    "_type": data["color"],
                    "_cutDirection": data["cut_dir"]
                })
            elif event["type"] == "bomb":
                notes.append({
                    "_time": beat_time,
                    "_lineIndex": data["lane"],
                    "_lineLayer": data["row"],
                    "_type": 3,  # Bomb type
                    "_cutDirection": 0
                })
            elif event["type"] == "obstacle":
                duration_beats = data.get("duration_ticks", 16) / ticks_per_beat
                obstacles.append({
                    "_time": beat_time,
                    "_lineIndex": data["lane"],
                    "_type": 0,  # Wall
                    "_duration": duration_beats,
                    "_width": data.get("width", 1)
                })

        # Sort by time
        notes.sort(key=lambda n: n["_time"])
        obstacles.sort(key=lambda o: o["_time"])

        difficulty_data = {
            "_version": "2.0.0",
            "_notes": notes,
            "_obstacles": obstacles,
            "_events": [],
            "_waypoints": []
        }

        filename = f"{difficulty}Standard.dat"
        dat_path = output_dir / filename

        with open(dat_path, "w", encoding="utf-8") as f:
            json.dump(difficulty_data, f)

        logger.info(f"Wrote {dat_path} ({len(notes)} notes, {len(obstacles)} obstacles)")


def main():
    parser = argparse.ArgumentParser(description="Generate Beat Saber map from audio")

    # Required
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--output", required=True, help="Output directory")

    # Model
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Model checkpoint")
    parser.add_argument("--model-size", choices=["small", "base", "large"], default="small")

    # Generation settings
    parser.add_argument("--difficulty", default="Expert",
                       choices=["Easy", "Normal", "Hard", "Expert", "ExpertPlus", "Expert+"])
    parser.add_argument("--bpm", type=float, required=True, help="Song BPM")
    parser.add_argument("--song-name", help="Song name (default: filename)")
    parser.add_argument("--song-author", help="Song author")
    parser.add_argument("--level-author", default="BeatSaber-ML", help="Level author")

    # Sampling
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--max-length", type=int, default=4000, help="Max tokens")
    parser.add_argument("--candidates", type=int, default=1, help="Number of candidates")

    # Hardware
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")

    args = parser.parse_args()

    # Load generator
    generator = MapGenerator.from_checkpoint(
        args.checkpoint,
        model_size=args.model_size,
        device=args.device
    )

    # Generate
    output_path = generator.generate(
        audio_path=args.audio,
        output_dir=args.output,
        difficulty=args.difficulty,
        bpm=args.bpm,
        song_name=args.song_name,
        song_author=args.song_author,
        level_author=args.level_author,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        validate=not args.no_validate,
        n_candidates=args.candidates
    )

    print(f"\nMap generated: {output_path}")
    print("\nTo play in Beat Saber:")
    print(f"  1. Copy '{output_path}' to your CustomLevels folder")
    print("  2. Refresh songs in Beat Saber")
    print("  3. Find the song and play!")


if __name__ == "__main__":
    main()
