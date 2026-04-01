"""
Audio Feature Extraction for Beat Saber ML

Extracts mel spectrograms and onset features aligned to a tick grid.

Features per tick:
- Mel spectrogram context window (128 bins × context frames)
- Onset strength
- RMS energy

Output shape: [n_ticks, feature_dim]
"""

import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import librosa
except ImportError:
    raise ImportError("librosa is required. Install with: pip install librosa")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """Container for extracted audio features."""
    features: np.ndarray      # [n_ticks, feature_dim]
    tick_times: np.ndarray    # [n_ticks] in seconds
    sample_rate: int
    bpm: float
    duration: float           # Audio duration in seconds

    @property
    def n_ticks(self) -> int:
        return len(self.tick_times)

    @property
    def feature_dim(self) -> int:
        return self.features.shape[1] if len(self.features.shape) > 1 else 0


class AudioFeatureExtractor:
    """
    Extracts audio features aligned to a tick grid for Beat Saber ML.

    The tick grid is defined by BPM and ticks_per_beat:
    - At 120 BPM with 16 ticks/beat, each tick = 1/32 second
    - Features are extracted at each tick position

    Features include:
    - Mel spectrogram with temporal context (past and future frames)
    - Onset strength
    - RMS energy

    Example:
        extractor = AudioFeatureExtractor(n_mels=128, context_frames=5)
        features = extractor.extract("song.egg", bpm=120, ticks_per_beat=16)
        print(f"Shape: {features.features.shape}")  # [n_ticks, 1408]
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        context_frames: int = 5,
        include_onset: bool = True,
        include_energy: bool = True,
        normalize: bool = True
    ):
        """
        Args:
            sample_rate: Target sample rate for audio loading
            n_mels: Number of mel frequency bins
            n_fft: FFT window size
            hop_length: Hop length for STFT
            context_frames: Number of frames before and after each tick
                           Total context = 2 * context_frames + 1
            include_onset: Include onset strength feature
            include_energy: Include RMS energy feature
            normalize: Normalize features to zero mean, unit variance
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.context_frames = context_frames
        self.include_onset = include_onset
        self.include_energy = include_energy
        self.normalize = normalize

        # Calculate feature dimension
        self.context_width = 2 * context_frames + 1
        self.mel_feature_dim = n_mels * self.context_width

        extra_features = 0
        if include_onset:
            extra_features += self.context_width
        if include_energy:
            extra_features += self.context_width

        self.feature_dim = self.mel_feature_dim + extra_features

    def extract(
        self,
        audio_path: Path,
        bpm: float,
        ticks_per_beat: int = 16,
        duration: Optional[float] = None
    ) -> AudioFeatures:
        """
        Extract features from an audio file.

        Args:
            audio_path: Path to audio file (.ogg, .egg, .mp3, .wav)
            bpm: Beats per minute of the song
            ticks_per_beat: Number of ticks per beat (default 16 = 1/16 beat grid)
            duration: Optional duration limit in seconds

        Returns:
            AudioFeatures object containing features and metadata
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        logger.debug(f"Loading audio: {audio_path}")
        y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=duration)
        audio_duration = len(y) / sr

        # Compute mel spectrogram
        logger.debug("Computing mel spectrogram...")
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Compute onset strength
        onset_env = None
        if self.include_onset:
            logger.debug("Computing onset strength...")
            onset_env = librosa.onset.onset_strength(
                y=y, sr=sr, hop_length=self.hop_length
            )

        # Compute RMS energy
        rms = None
        if self.include_energy:
            logger.debug("Computing RMS energy...")
            rms = librosa.feature.rms(
                y=y, hop_length=self.hop_length
            )[0]

        # Generate tick times
        tick_times = self._generate_tick_times(bpm, ticks_per_beat, audio_duration)
        n_ticks = len(tick_times)

        logger.debug(f"Extracting features for {n_ticks} ticks...")

        # Get frame times for alignment
        frame_times = librosa.frames_to_time(
            np.arange(mel_db.shape[1]),
            sr=sr,
            hop_length=self.hop_length
        )

        # Extract features for each tick
        features = np.zeros((n_ticks, self.feature_dim), dtype=np.float32)

        for i, tick_time in enumerate(tick_times):
            # Find nearest frame
            frame_idx = np.argmin(np.abs(frame_times - tick_time))

            # Extract mel context window
            mel_context = self._extract_context(mel_db, frame_idx)

            # Build feature vector
            feat_idx = 0

            # Mel features (flattened context window)
            features[i, feat_idx:feat_idx + self.mel_feature_dim] = mel_context.flatten()
            feat_idx += self.mel_feature_dim

            # Onset features
            if self.include_onset and onset_env is not None:
                onset_context = self._extract_context_1d(onset_env, frame_idx)
                features[i, feat_idx:feat_idx + self.context_width] = onset_context
                feat_idx += self.context_width

            # Energy features
            if self.include_energy and rms is not None:
                rms_context = self._extract_context_1d(rms, frame_idx)
                features[i, feat_idx:feat_idx + self.context_width] = rms_context
                feat_idx += self.context_width

        # Normalize features
        if self.normalize:
            features = self._normalize_features(features)

        return AudioFeatures(
            features=features,
            tick_times=tick_times,
            sample_rate=sr,
            bpm=bpm,
            duration=audio_duration
        )

    def _generate_tick_times(
        self,
        bpm: float,
        ticks_per_beat: int,
        duration: float
    ) -> np.ndarray:
        """Generate tick times in seconds."""
        seconds_per_tick = 60.0 / (bpm * ticks_per_beat)
        n_ticks = int(duration / seconds_per_tick) + 1
        return np.arange(n_ticks) * seconds_per_tick

    def _extract_context(self, spectrogram: np.ndarray, center_frame: int) -> np.ndarray:
        """Extract context window around a frame from 2D spectrogram."""
        n_bins, n_frames = spectrogram.shape

        start = center_frame - self.context_frames
        end = center_frame + self.context_frames + 1

        # Handle edge cases with padding
        if start < 0 or end > n_frames:
            # Create padded window
            context = np.zeros((n_bins, self.context_width), dtype=spectrogram.dtype)

            # Calculate valid range
            src_start = max(0, start)
            src_end = min(n_frames, end)
            dst_start = max(0, -start)
            dst_end = dst_start + (src_end - src_start)

            context[:, dst_start:dst_end] = spectrogram[:, src_start:src_end]

            # Edge padding
            if dst_start > 0:
                context[:, :dst_start] = spectrogram[:, 0:1]
            if dst_end < self.context_width:
                context[:, dst_end:] = spectrogram[:, -1:]
        else:
            context = spectrogram[:, start:end]

        return context

    def _extract_context_1d(self, signal: np.ndarray, center_frame: int) -> np.ndarray:
        """Extract context window from 1D signal."""
        n_frames = len(signal)

        start = center_frame - self.context_frames
        end = center_frame + self.context_frames + 1

        if start < 0 or end > n_frames:
            context = np.zeros(self.context_width, dtype=signal.dtype)

            src_start = max(0, start)
            src_end = min(n_frames, end)
            dst_start = max(0, -start)
            dst_end = dst_start + (src_end - src_start)

            context[dst_start:dst_end] = signal[src_start:src_end]

            if dst_start > 0:
                context[:dst_start] = signal[0]
            if dst_end < self.context_width:
                context[dst_end:] = signal[-1]
        else:
            context = signal[start:end]

        return context

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to zero mean and unit variance."""
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0  # Avoid division by zero
        return (features - mean) / std


def extract_audio_features(
    audio_path: Path,
    bpm: float,
    ticks_per_beat: int = 16,
    **kwargs
) -> AudioFeatures:
    """
    Convenience function to extract audio features.

    Args:
        audio_path: Path to audio file
        bpm: Beats per minute
        ticks_per_beat: Ticks per beat (default 16)
        **kwargs: Additional arguments for AudioFeatureExtractor

    Returns:
        AudioFeatures object
    """
    extractor = AudioFeatureExtractor(**kwargs)
    return extractor.extract(audio_path, bpm, ticks_per_beat)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract audio features')
    parser.add_argument('audio', help='Path to audio file')
    parser.add_argument('--bpm', type=float, required=True, help='BPM of the song')
    parser.add_argument('--ticks', type=int, default=16, help='Ticks per beat')
    parser.add_argument('--context', type=int, default=5, help='Context frames')
    parser.add_argument('--mels', type=int, default=128, help='Mel bins')

    args = parser.parse_args()

    extractor = AudioFeatureExtractor(
        n_mels=args.mels,
        context_frames=args.context
    )

    features = extractor.extract(
        Path(args.audio),
        bpm=args.bpm,
        ticks_per_beat=args.ticks
    )

    print(f"\n=== Audio Features ===")
    print(f"Audio duration: {features.duration:.1f}s")
    print(f"BPM: {features.bpm}")
    print(f"Ticks: {features.n_ticks}")
    print(f"Feature shape: {features.features.shape}")
    print(f"Feature dim: {features.feature_dim}")
    print(f"Tick times: {features.tick_times[:5]}... (first 5)")
    print(f"Feature stats: mean={features.features.mean():.3f}, std={features.features.std():.3f}")
