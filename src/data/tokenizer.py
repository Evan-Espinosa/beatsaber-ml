"""
Event Tokenizer for Beat Saber ML

Converts canonical map events to discrete token sequences and back.

Token Vocabulary (~407 tokens):
- Special tokens: BOS, EOS, PAD (3)
- TIME_SHIFT_1 through TIME_SHIFT_64 (64)
- NOTE_{lane}_{row}_{color}_{dir} (4 * 3 * 2 * 9 = 216)
- BOMB_{lane}_{row} (4 * 3 = 12)
- OBST_WALL_{lane}_{width}_{durbin} (4 * 4 * 7 = 112)

Total: 3 + 64 + 216 + 12 + 112 = 407 tokens
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Grid dimensions
NUM_LANES = 4    # Columns 0-3
NUM_ROWS = 3     # Rows 0-2
NUM_COLORS = 2   # 0=red, 1=blue
NUM_DIRS = 9     # 0-8 cut directions

# Time shift constants
MAX_TIME_SHIFT = 64  # Maximum ticks per TIME_SHIFT token

# Obstacle duration bins (upper bounds in ticks)
DURATION_BINS = [8, 16, 32, 64, 128, 256, 999]

# Special tokens
BOS_TOKEN = "BOS"
EOS_TOKEN = "EOS"
PAD_TOKEN = "PAD"


@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer."""
    num_lanes: int = NUM_LANES
    num_rows: int = NUM_ROWS
    num_colors: int = NUM_COLORS
    num_directions: int = NUM_DIRS
    max_time_shift: int = MAX_TIME_SHIFT
    duration_bins: tuple = tuple(DURATION_BINS)


class EventTokenizer:
    """
    Converts Beat Saber events to/from token sequences.

    The tokenization scheme uses:
    - TIME_SHIFT tokens to advance time (in ticks)
    - NOTE tokens encoding position (lane, row), color, and cut direction
    - BOMB tokens encoding position only
    - OBSTACLE tokens encoding position, width, and duration bin

    Example:
        tokenizer = EventTokenizer()
        print(f"Vocabulary size: {tokenizer.vocab_size}")

        # Convert events to tokens
        tokens = tokenizer.encode(canonical_events)
        print(f"Tokens: {tokens[:10]}")

        # Convert back to events
        events = tokenizer.decode(tokens)
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        """
        Args:
            config: Optional tokenizer configuration
        """
        self.config = config or TokenizerConfig()
        self._build_vocabulary()

    def _build_vocabulary(self):
        """Build the token vocabulary."""
        vocab = []

        # Special tokens (indices 0, 1, 2)
        vocab.append(PAD_TOKEN)  # 0 - padding
        vocab.append(BOS_TOKEN)  # 1 - beginning of sequence
        vocab.append(EOS_TOKEN)  # 2 - end of sequence

        # TIME_SHIFT tokens (indices 3-66)
        for k in range(1, self.config.max_time_shift + 1):
            vocab.append(f"TIME_SHIFT_{k}")

        # NOTE tokens (indices 67-282)
        # NOTE_{lane}_{row}_{color}_{direction}
        for lane in range(self.config.num_lanes):
            for row in range(self.config.num_rows):
                for color in range(self.config.num_colors):
                    for direction in range(self.config.num_directions):
                        vocab.append(f"NOTE_{lane}_{row}_{color}_{direction}")

        # BOMB tokens (indices 283-294)
        for lane in range(self.config.num_lanes):
            for row in range(self.config.num_rows):
                vocab.append(f"BOMB_{lane}_{row}")

        # OBSTACLE tokens (indices 295-406)
        # OBST_WALL_{lane}_{width}_{duration_bin}
        for lane in range(self.config.num_lanes):
            for width in range(1, self.config.num_lanes + 1):
                for dur_bin in range(len(self.config.duration_bins)):
                    vocab.append(f"OBST_WALL_{lane}_{width}_{dur_bin}")

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        # Special token IDs
        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.bos_id = self.token_to_id[BOS_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]

    def encode(self, events: list, add_special_tokens: bool = True) -> list[int]:
        """
        Convert canonical events to token ID sequence.

        Args:
            events: List of canonical event dicts
            add_special_tokens: Add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.bos_id)

        # Sort events by tick
        sorted_events = sorted(events, key=lambda e: e["tick"])

        current_tick = 0

        for event in sorted_events:
            event_tick = event["tick"]

            # Emit TIME_SHIFT tokens
            tick_delta = event_tick - current_tick
            while tick_delta > 0:
                shift = min(tick_delta, self.config.max_time_shift)
                token = f"TIME_SHIFT_{shift}"
                tokens.append(self.token_to_id[token])
                tick_delta -= shift
                current_tick += shift

            # Emit event token
            event_token = self._event_to_token(event)
            if event_token and event_token in self.token_to_id:
                tokens.append(self.token_to_id[event_token])

        if add_special_tokens:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, token_ids: list[int], include_special: bool = False) -> list[dict]:
        """
        Convert token ID sequence back to canonical events.

        Args:
            token_ids: List of token IDs
            include_special: If False, skip BOS/EOS/PAD tokens

        Returns:
            List of canonical event dicts
        """
        events = []
        current_tick = 0

        for token_id in token_ids:
            if token_id >= len(self.id_to_token):
                continue

            token = self.id_to_token[token_id]

            # Skip special tokens
            if not include_special and token in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
                continue

            # Handle TIME_SHIFT
            if token.startswith("TIME_SHIFT_"):
                shift = int(token.split("_")[2])
                current_tick += shift
                continue

            # Parse event token
            event = self._token_to_event(token, current_tick)
            if event:
                events.append(event)

        return events

    def _event_to_token(self, event: dict) -> Optional[str]:
        """Convert a single canonical event to a token string."""
        event_type = event["type"]
        data = event["data"]

        if event_type == "note":
            lane = self._clamp(data["lane"], 0, self.config.num_lanes - 1)
            row = self._clamp(data["row"], 0, self.config.num_rows - 1)
            color = self._clamp(data["color"], 0, self.config.num_colors - 1)
            direction = self._clamp(data["cut_dir"], 0, self.config.num_directions - 1)
            return f"NOTE_{lane}_{row}_{color}_{direction}"

        elif event_type == "bomb":
            lane = self._clamp(data["lane"], 0, self.config.num_lanes - 1)
            row = self._clamp(data["row"], 0, self.config.num_rows - 1)
            return f"BOMB_{lane}_{row}"

        elif event_type == "obstacle":
            lane = self._clamp(data["lane"], 0, self.config.num_lanes - 1)
            width = self._clamp(data.get("width", 1), 1, self.config.num_lanes)
            duration_ticks = data.get("duration_ticks", 16)
            dur_bin = self._duration_to_bin(duration_ticks)
            return f"OBST_WALL_{lane}_{width}_{dur_bin}"

        return None

    def _token_to_event(self, token: str, tick: int) -> Optional[dict]:
        """Convert a token string back to a canonical event."""
        # Parse NOTE token
        if token.startswith("NOTE_"):
            parts = token.split("_")
            if len(parts) == 5:
                lane, row, color, direction = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                return {
                    "tick": tick,
                    "t_sec": None,  # Will be filled by caller if needed
                    "type": "note",
                    "data": {
                        "lane": lane,
                        "row": row,
                        "color": color,
                        "cut_dir": direction
                    }
                }

        # Parse BOMB token
        elif token.startswith("BOMB_"):
            parts = token.split("_")
            if len(parts) == 3:
                lane, row = int(parts[1]), int(parts[2])
                return {
                    "tick": tick,
                    "t_sec": None,
                    "type": "bomb",
                    "data": {
                        "lane": lane,
                        "row": row
                    }
                }

        # Parse OBSTACLE token
        elif token.startswith("OBST_WALL_"):
            parts = token.split("_")
            if len(parts) == 5:
                lane, width, dur_bin = int(parts[2]), int(parts[3]), int(parts[4])
                duration_ticks = self._bin_to_duration(dur_bin)
                return {
                    "tick": tick,
                    "t_sec": None,
                    "type": "obstacle",
                    "data": {
                        "lane": lane,
                        "width": width,
                        "height": 5,  # Default full height
                        "duration_ticks": duration_ticks,
                        "wall_type": "wall"
                    }
                }

        return None

    def _duration_to_bin(self, duration_ticks: int) -> int:
        """Convert duration in ticks to bin index."""
        for i, upper in enumerate(self.config.duration_bins):
            if duration_ticks <= upper:
                return i
        return len(self.config.duration_bins) - 1

    def _bin_to_duration(self, bin_idx: int) -> int:
        """Convert bin index back to representative duration."""
        if bin_idx == 0:
            return 4  # Small obstacle
        elif bin_idx < len(self.config.duration_bins):
            # Return midpoint of bin
            lower = self.config.duration_bins[bin_idx - 1] if bin_idx > 0 else 0
            upper = self.config.duration_bins[bin_idx]
            return (lower + upper) // 2
        return 256  # Large obstacle

    def _clamp(self, value: int, min_val: int, max_val: int) -> int:
        """Clamp value to range."""
        return max(min_val, min(max_val, value))

    def tokens_to_string(self, token_ids: list[int]) -> list[str]:
        """Convert token IDs to token strings."""
        return [self.id_to_token.get(tid, f"UNK_{tid}") for tid in token_ids]

    def string_to_tokens(self, token_strings: list[str]) -> list[int]:
        """Convert token strings to token IDs."""
        return [self.token_to_id.get(s, self.pad_id) for s in token_strings]


def verify_roundtrip(events: list, tokenizer: EventTokenizer) -> bool:
    """
    Verify that events can be encoded and decoded without loss.

    Args:
        events: List of canonical events
        tokenizer: EventTokenizer instance

    Returns:
        True if roundtrip successful
    """
    # Encode
    token_ids = tokenizer.encode(events)

    # Decode
    decoded_events = tokenizer.decode(token_ids)

    # Compare (ignoring t_sec which isn't preserved)
    if len(events) != len(decoded_events):
        logger.warning(f"Event count mismatch: {len(events)} vs {len(decoded_events)}")
        return False

    for orig, decoded in zip(sorted(events, key=lambda e: e["tick"]),
                              sorted(decoded_events, key=lambda e: e["tick"])):
        if orig["tick"] != decoded["tick"]:
            logger.warning(f"Tick mismatch: {orig['tick']} vs {decoded['tick']}")
            return False
        if orig["type"] != decoded["type"]:
            logger.warning(f"Type mismatch: {orig['type']} vs {decoded['type']}")
            return False
        # Compare relevant data fields
        for key in ["lane", "row", "color", "cut_dir"]:
            if key in orig["data"] and orig["data"][key] != decoded["data"].get(key):
                logger.warning(f"Data mismatch for {key}: {orig['data'][key]} vs {decoded['data'].get(key)}")
                return False

    return True


# CLI interface
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from src.data.parser import parse_map

    arg_parser = argparse.ArgumentParser(description='Tokenize Beat Saber maps')
    arg_parser.add_argument('--map', help='Map folder to tokenize')
    arg_parser.add_argument('--verify', action='store_true', help='Verify roundtrip')

    args = arg_parser.parse_args()

    tokenizer = EventTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: PAD={tokenizer.pad_id}, BOS={tokenizer.bos_id}, EOS={tokenizer.eos_id}")
    print(f"\nSample tokens:")
    print(f"  TIME_SHIFT_1 = {tokenizer.token_to_id['TIME_SHIFT_1']}")
    print(f"  TIME_SHIFT_64 = {tokenizer.token_to_id['TIME_SHIFT_64']}")
    print(f"  NOTE_0_0_0_0 = {tokenizer.token_to_id['NOTE_0_0_0_0']}")
    print(f"  NOTE_3_2_1_8 = {tokenizer.token_to_id['NOTE_3_2_1_8']}")
    print(f"  BOMB_2_1 = {tokenizer.token_to_id['BOMB_2_1']}")
    print(f"  OBST_WALL_0_1_0 = {tokenizer.token_to_id['OBST_WALL_0_1_0']}")

    if args.map:
        maps = parse_map(Path(args.map))

        for m in maps:
            print(f"\n=== {m.difficulty.name} ===")
            print(f"Events: {len(m.events)}")

            token_ids = tokenizer.encode(m.events)
            print(f"Tokens: {len(token_ids)}")
            print(f"First 15 tokens: {tokenizer.tokens_to_string(token_ids[:15])}")

            if args.verify:
                success = verify_roundtrip(m.events, tokenizer)
                print(f"Roundtrip verification: {'PASS' if success else 'FAIL'}")
