"""
Beat Saber Map Parser

Parses v2 and v3 Beat Saber map formats to a unified canonical schema.

Canonical Event Format:
{
    "t_sec": float,      # Time in seconds
    "tick": int,         # Time in ticks (1/16 beat grid)
    "type": str,         # "note", "bomb", "obstacle"
    "data": {
        # For notes:
        "lane": int,     # 0-3 (left to right)
        "row": int,      # 0-2 (bottom to top)
        "color": int,    # 0=red, 1=blue
        "cut_dir": int,  # 0-8 (see CUT_DIRECTIONS)

        # For obstacles:
        "lane": int,
        "width": int,
        "height": int,
        "duration_ticks": int,
        "wall_type": str  # "wall" or "ceiling"
    }
}
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cut direction mapping (same for v2 and v3)
CUT_DIRECTIONS = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
    4: "up_left",
    5: "up_right",
    6: "down_left",
    7: "down_right",
    8: "dot"  # Any direction
}


@dataclass
class MapMetadata:
    """Metadata parsed from Info.dat"""
    song_name: str
    song_author: str
    level_author: str
    bpm: float
    song_offset: float  # seconds
    song_filename: str

    # BPM changes (if any)
    bpm_events: list = field(default_factory=list)

    # Available difficulties
    difficulties: list = field(default_factory=list)

    @property
    def has_bpm_changes(self) -> bool:
        """Check if map has BPM changes (complicates timing)."""
        return len(self.bpm_events) > 0

    @property
    def seconds_per_beat(self) -> float:
        """Seconds per beat at base BPM."""
        return 60.0 / self.bpm


@dataclass
class DifficultyInfo:
    """Info about a single difficulty."""
    name: str              # "Easy", "Normal", "Hard", "Expert", "ExpertPlus"
    rank: int              # 1, 3, 5, 7, 9
    filename: str          # "ExpertStandard.dat"
    characteristic: str    # "Standard", "OneSaber", etc.
    njs: float            # Note jump speed
    offset: float         # Note jump offset


@dataclass
class CanonicalMap:
    """Canonical representation of a Beat Saber map."""
    metadata: MapMetadata
    difficulty: DifficultyInfo
    events: list  # List of canonical events
    format_version: str  # "v2" or "v3"

    def __len__(self):
        return len(self.events)

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds (from last event)."""
        if not self.events:
            return 0.0
        return self.events[-1]["t_sec"]

    @property
    def note_count(self) -> int:
        return sum(1 for e in self.events if e["type"] == "note")

    @property
    def bomb_count(self) -> int:
        return sum(1 for e in self.events if e["type"] == "bomb")

    @property
    def obstacle_count(self) -> int:
        return sum(1 for e in self.events if e["type"] == "obstacle")


class BeatSaberParser:
    """
    Parses Beat Saber v2 and v3 map formats to canonical schema.

    Usage:
        parser = BeatSaberParser(ticks_per_beat=16)
        maps = parser.parse_map_folder(Path("data/raw/abc123"))

        for canonical_map in maps:
            print(f"{canonical_map.difficulty.name}: {canonical_map.note_count} notes")
    """

    def __init__(self, ticks_per_beat: int = 16):
        """
        Args:
            ticks_per_beat: Quantization resolution (16 = 1/16 beat grid)
        """
        self.ticks_per_beat = ticks_per_beat

    def parse_map_folder(self, folder: Path) -> list[CanonicalMap]:
        """
        Parse all difficulties in a map folder.

        Args:
            folder: Path to extracted map folder

        Returns:
            List of CanonicalMap objects (one per difficulty)
        """
        folder = Path(folder)

        # Find Info.dat (case-insensitive)
        info_path = None
        for name in ["Info.dat", "info.dat", "INFO.DAT"]:
            if (folder / name).exists():
                info_path = folder / name
                break

        if not info_path:
            raise FileNotFoundError(f"No Info.dat found in {folder}")

        # Parse Info.dat
        metadata = self._parse_info_dat(info_path)

        # Parse each difficulty
        results = []
        for diff_info in metadata.difficulties:
            diff_path = folder / diff_info.filename

            if not diff_path.exists():
                logger.warning(f"Difficulty file not found: {diff_path}")
                continue

            try:
                canonical = self._parse_difficulty(diff_path, metadata, diff_info)
                results.append(canonical)
            except Exception as e:
                logger.warning(f"Failed to parse {diff_path}: {e}")
                continue

        return results

    def _parse_info_dat(self, path: Path) -> MapMetadata:
        """Parse Info.dat file."""
        with open(path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        # Extract basic metadata
        metadata = MapMetadata(
            song_name=data.get('_songName', ''),
            song_author=data.get('_songAuthorName', ''),
            level_author=data.get('_levelAuthorName', ''),
            bpm=float(data.get('_beatsPerMinute', 120)),
            song_offset=float(data.get('_songTimeOffset', 0)),
            song_filename=data.get('_songFilename', 'song.egg'),
        )

        # Extract difficulties
        for beatmap_set in data.get('_difficultyBeatmapSets', []):
            characteristic = beatmap_set.get('_beatmapCharacteristicName', 'Standard')

            for diff in beatmap_set.get('_difficultyBeatmaps', []):
                diff_info = DifficultyInfo(
                    name=diff.get('_difficulty', 'Expert'),
                    rank=diff.get('_difficultyRank', 7),
                    filename=diff.get('_beatmapFilename', ''),
                    characteristic=characteristic,
                    njs=float(diff.get('_noteJumpMovementSpeed', 10)),
                    offset=float(diff.get('_noteJumpStartBeatOffset', 0))
                )
                metadata.difficulties.append(diff_info)

        return metadata

    def _parse_difficulty(
        self,
        path: Path,
        metadata: MapMetadata,
        diff_info: DifficultyInfo
    ) -> CanonicalMap:
        """Parse a difficulty file (v2 or v3)."""
        with open(path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        # Detect format version
        if '_notes' in data:
            format_version = 'v2'
            events = self._parse_v2_events(data, metadata)
        elif 'colorNotes' in data:
            format_version = 'v3'
            events = self._parse_v3_events(data, metadata)
        else:
            raise ValueError(f"Unknown map format in {path}")

        # Check for BPM changes
        bpm_events = data.get('_BPMChanges', data.get('bpmEvents', []))
        if bpm_events:
            metadata.bpm_events = bpm_events
            logger.info(f"Map has {len(bpm_events)} BPM changes")

        # Sort events by tick
        events.sort(key=lambda e: (e['tick'], e['type']))

        return CanonicalMap(
            metadata=metadata,
            difficulty=diff_info,
            events=events,
            format_version=format_version
        )

    def _parse_v2_events(self, data: dict, metadata: MapMetadata) -> list:
        """Parse v2 format events."""
        events = []

        # Parse notes
        for note in data.get('_notes', []):
            beat_time = float(note.get('_time', 0))
            note_type = int(note.get('_type', 0))

            # Skip non-note types
            if note_type == 3:
                # Bomb
                events.append(self._create_event(
                    beat_time=beat_time,
                    bpm=metadata.bpm,
                    event_type="bomb",
                    data={
                        "lane": int(note.get('_lineIndex', 0)),
                        "row": int(note.get('_lineLayer', 0))
                    }
                ))
            elif note_type in (0, 1):
                # Regular note (0=red, 1=blue)
                events.append(self._create_event(
                    beat_time=beat_time,
                    bpm=metadata.bpm,
                    event_type="note",
                    data={
                        "lane": int(note.get('_lineIndex', 0)),
                        "row": int(note.get('_lineLayer', 0)),
                        "color": note_type,
                        "cut_dir": int(note.get('_cutDirection', 0))
                    }
                ))

        # Parse obstacles
        for obs in data.get('_obstacles', []):
            beat_time = float(obs.get('_time', 0))
            duration_beats = float(obs.get('_duration', 0))
            duration_ticks = self._beats_to_ticks(duration_beats)

            obs_type = int(obs.get('_type', 0))
            wall_type = "ceiling" if obs_type == 1 else "wall"

            events.append(self._create_event(
                beat_time=beat_time,
                bpm=metadata.bpm,
                event_type="obstacle",
                data={
                    "lane": int(obs.get('_lineIndex', 0)),
                    "width": int(obs.get('_width', 1)),
                    "height": 5 if obs_type == 0 else 1,  # Full height or ceiling
                    "duration_ticks": duration_ticks,
                    "wall_type": wall_type
                }
            ))

        return events

    def _parse_v3_events(self, data: dict, metadata: MapMetadata) -> list:
        """Parse v3 format events."""
        events = []

        # Parse color notes
        for note in data.get('colorNotes', []):
            beat_time = float(note.get('b', 0))

            events.append(self._create_event(
                beat_time=beat_time,
                bpm=metadata.bpm,
                event_type="note",
                data={
                    "lane": int(note.get('x', 0)),
                    "row": int(note.get('y', 0)),
                    "color": int(note.get('c', 0)),
                    "cut_dir": int(note.get('d', 0))
                }
            ))

        # Parse bombs
        for bomb in data.get('bombNotes', []):
            beat_time = float(bomb.get('b', 0))

            events.append(self._create_event(
                beat_time=beat_time,
                bpm=metadata.bpm,
                event_type="bomb",
                data={
                    "lane": int(bomb.get('x', 0)),
                    "row": int(bomb.get('y', 0))
                }
            ))

        # Parse obstacles
        for obs in data.get('obstacles', []):
            beat_time = float(obs.get('b', 0))
            duration_beats = float(obs.get('d', 0))
            duration_ticks = self._beats_to_ticks(duration_beats)

            events.append(self._create_event(
                beat_time=beat_time,
                bpm=metadata.bpm,
                event_type="obstacle",
                data={
                    "lane": int(obs.get('x', 0)),
                    "width": int(obs.get('w', 1)),
                    "height": int(obs.get('h', 5)),
                    "duration_ticks": duration_ticks,
                    "wall_type": "wall"
                }
            ))

        return events

    def _create_event(
        self,
        beat_time: float,
        bpm: float,
        event_type: str,
        data: dict
    ) -> dict:
        """Create a canonical event with timing info."""
        t_sec = self._beats_to_seconds(beat_time, bpm)
        tick = self._beats_to_ticks(beat_time)

        return {
            "t_sec": t_sec,
            "tick": tick,
            "type": event_type,
            "data": data
        }

    def _beats_to_seconds(self, beats: float, bpm: float) -> float:
        """Convert beat time to seconds."""
        return beats * 60.0 / bpm

    def _beats_to_ticks(self, beats: float) -> int:
        """Convert beat time to tick grid."""
        return round(beats * self.ticks_per_beat)

    def _ticks_to_beats(self, ticks: int) -> float:
        """Convert ticks back to beats."""
        return ticks / self.ticks_per_beat


def parse_map(folder: Path, ticks_per_beat: int = 16) -> list[CanonicalMap]:
    """
    Convenience function to parse a map folder.

    Args:
        folder: Path to map folder
        ticks_per_beat: Quantization resolution

    Returns:
        List of CanonicalMap objects
    """
    parser = BeatSaberParser(ticks_per_beat=ticks_per_beat)
    return parser.parse_map_folder(folder)


# CLI interface
if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description='Parse Beat Saber maps')
    arg_parser.add_argument('folder', help='Map folder to parse')
    arg_parser.add_argument('--ticks', type=int, default=16, help='Ticks per beat')
    arg_parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = arg_parser.parse_args()

    maps = parse_map(Path(args.folder), ticks_per_beat=args.ticks)

    for m in maps:
        print(f"\n=== {m.difficulty.characteristic} - {m.difficulty.name} ===")
        print(f"Format: {m.format_version}")
        print(f"BPM: {m.metadata.bpm}")
        print(f"Duration: {m.duration_seconds:.1f}s")
        print(f"Notes: {m.note_count}")
        print(f"Bombs: {m.bomb_count}")
        print(f"Obstacles: {m.obstacle_count}")
        print(f"BPM Changes: {m.metadata.has_bpm_changes}")

        if args.json and m.events:
            print("\nFirst 5 events:")
            for e in m.events[:5]:
                print(f"  {json.dumps(e)}")
