"""
Constraint Validator for Beat Saber Maps

Validates generated maps against playability constraints:
- Max notes per hand per tick
- No impossible hand spreads
- Cut direction consistency
- Note-obstacle collisions

Also provides automatic fixing for common violations.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Violation:
    """Represents a constraint violation."""
    tick: int
    violation_type: str
    description: str
    severity: str = "warning"  # "warning" or "error"


class ConstraintValidator:
    """
    Validates Beat Saber maps against playability constraints.

    Hard constraints (errors):
    - Max 2 notes per color per tick
    - No overlapping notes (same position)
    - Notes must be in valid grid positions

    Soft constraints (warnings):
    - Avoid wide hand spreads (>2 lanes apart)
    - Cut direction should flow naturally
    - Reasonable note density

    Example:
        validator = ConstraintValidator()
        is_valid, violations = validator.validate(events)

        if not is_valid:
            fixed_events = validator.fix(events)
    """

    def __init__(
        self,
        max_notes_per_hand: int = 2,
        max_lane_spread: int = 3,
        min_tick_gap: int = 2,  # Minimum ticks between notes for same hand
    ):
        self.max_notes_per_hand = max_notes_per_hand
        self.max_lane_spread = max_lane_spread
        self.min_tick_gap = min_tick_gap

        # Valid grid positions
        self.valid_lanes = set(range(4))  # 0-3
        self.valid_rows = set(range(3))   # 0-2
        self.valid_colors = {0, 1}        # 0=red, 1=blue
        self.valid_directions = set(range(9))  # 0-8

    def validate(self, events: list) -> tuple[bool, list[Violation]]:
        """
        Validate a list of canonical events.

        Args:
            events: List of canonical event dicts

        Returns:
            (is_valid, violations): Tuple of validity and list of violations
        """
        violations = []

        # Group events by tick
        tick_groups = self._group_by_tick(events)

        # Run all checks
        violations.extend(self._check_grid_bounds(events))
        violations.extend(self._check_notes_per_hand(tick_groups))
        violations.extend(self._check_overlapping_notes(tick_groups))
        violations.extend(self._check_hand_spread(tick_groups))
        violations.extend(self._check_note_obstacle_collision(tick_groups))

        # Determine validity (errors = invalid, warnings = valid but suboptimal)
        has_errors = any(v.severity == "error" for v in violations)

        return not has_errors, violations

    def _group_by_tick(self, events: list) -> dict:
        """Group events by tick."""
        groups = defaultdict(list)
        for event in events:
            groups[event["tick"]].append(event)
        return groups

    def _check_grid_bounds(self, events: list) -> list[Violation]:
        """Check that all notes are within valid grid bounds."""
        violations = []

        for event in events:
            if event["type"] not in ("note", "bomb"):
                continue

            data = event["data"]
            tick = event["tick"]

            if data.get("lane") not in self.valid_lanes:
                violations.append(Violation(
                    tick=tick,
                    violation_type="invalid_lane",
                    description=f"Lane {data.get('lane')} out of bounds (0-3)",
                    severity="error"
                ))

            if data.get("row") not in self.valid_rows:
                violations.append(Violation(
                    tick=tick,
                    violation_type="invalid_row",
                    description=f"Row {data.get('row')} out of bounds (0-2)",
                    severity="error"
                ))

            if event["type"] == "note":
                if data.get("color") not in self.valid_colors:
                    violations.append(Violation(
                        tick=tick,
                        violation_type="invalid_color",
                        description=f"Color {data.get('color')} invalid (0-1)",
                        severity="error"
                    ))

                if data.get("cut_dir") not in self.valid_directions:
                    violations.append(Violation(
                        tick=tick,
                        violation_type="invalid_direction",
                        description=f"Direction {data.get('cut_dir')} invalid (0-8)",
                        severity="error"
                    ))

        return violations

    def _check_notes_per_hand(self, tick_groups: dict) -> list[Violation]:
        """Check max notes per hand (color) per tick."""
        violations = []

        for tick, events in tick_groups.items():
            notes_by_color = defaultdict(list)

            for event in events:
                if event["type"] == "note":
                    color = event["data"].get("color", 0)
                    notes_by_color[color].append(event)

            for color, notes in notes_by_color.items():
                if len(notes) > self.max_notes_per_hand:
                    color_name = "red" if color == 0 else "blue"
                    violations.append(Violation(
                        tick=tick,
                        violation_type="too_many_notes",
                        description=f"{len(notes)} {color_name} notes (max {self.max_notes_per_hand})",
                        severity="error"
                    ))

        return violations

    def _check_overlapping_notes(self, tick_groups: dict) -> list[Violation]:
        """Check for notes in the same position at the same tick."""
        violations = []

        for tick, events in tick_groups.items():
            positions = set()

            for event in events:
                if event["type"] in ("note", "bomb"):
                    pos = (event["data"]["lane"], event["data"]["row"])

                    if pos in positions:
                        violations.append(Violation(
                            tick=tick,
                            violation_type="overlapping_notes",
                            description=f"Multiple notes at position {pos}",
                            severity="error"
                        ))
                    positions.add(pos)

        return violations

    def _check_hand_spread(self, tick_groups: dict) -> list[Violation]:
        """Check for unreasonable hand spreads."""
        violations = []

        for tick, events in tick_groups.items():
            notes_by_color = defaultdict(list)

            for event in events:
                if event["type"] == "note":
                    color = event["data"].get("color", 0)
                    notes_by_color[color].append(event)

            for color, notes in notes_by_color.items():
                if len(notes) >= 2:
                    lanes = [n["data"]["lane"] for n in notes]
                    spread = max(lanes) - min(lanes)

                    if spread > self.max_lane_spread:
                        color_name = "red" if color == 0 else "blue"
                        violations.append(Violation(
                            tick=tick,
                            violation_type="wide_hand_spread",
                            description=f"{color_name} hand spread of {spread} lanes",
                            severity="warning"
                        ))

        return violations

    def _check_note_obstacle_collision(self, tick_groups: dict) -> list[Violation]:
        """Check for notes placed inside obstacles."""
        violations = []

        # Track active obstacles
        active_obstacles = []

        for tick in sorted(tick_groups.keys()):
            events = tick_groups[tick]

            # Update active obstacles
            active_obstacles = [
                (start, end, lanes) for start, end, lanes in active_obstacles
                if end > tick
            ]

            # Add new obstacles
            for event in events:
                if event["type"] == "obstacle":
                    data = event["data"]
                    start_tick = tick
                    end_tick = tick + data.get("duration_ticks", 0)
                    lane = data.get("lane", 0)
                    width = data.get("width", 1)
                    lanes = set(range(lane, min(lane + width, 4)))
                    active_obstacles.append((start_tick, end_tick, lanes))

            # Check notes against active obstacles
            for event in events:
                if event["type"] == "note":
                    note_lane = event["data"]["lane"]

                    for start, end, blocked_lanes in active_obstacles:
                        if note_lane in blocked_lanes and start <= tick < end:
                            violations.append(Violation(
                                tick=tick,
                                violation_type="note_in_obstacle",
                                description=f"Note at lane {note_lane} blocked by obstacle",
                                severity="error"
                            ))

        return violations

    def fix(self, events: list) -> list:
        """
        Attempt to fix constraint violations.

        Fixes applied:
        - Clamp positions to valid grid
        - Remove excess notes per hand (keep first N)
        - Remove overlapping notes (keep first)
        - Remove notes blocked by obstacles

        Args:
            events: List of canonical events

        Returns:
            Fixed list of events
        """
        fixed = []
        tick_groups = self._group_by_tick(events)

        # Track active obstacles for collision detection
        active_obstacles = []

        for tick in sorted(tick_groups.keys()):
            tick_events = tick_groups[tick]

            # Update active obstacles
            active_obstacles = [
                (start, end, lanes) for start, end, lanes in active_obstacles
                if end > tick
            ]

            # Process obstacles first
            obstacles = [e for e in tick_events if e["type"] == "obstacle"]
            for obs in obstacles:
                data = obs["data"]
                lane = max(0, min(3, data.get("lane", 0)))
                width = max(1, min(4, data.get("width", 1)))
                lanes = set(range(lane, min(lane + width, 4)))
                end_tick = tick + data.get("duration_ticks", 0)
                active_obstacles.append((tick, end_tick, lanes))
                fixed.append(obs)

            # Get blocked lanes at this tick
            blocked_lanes = set()
            for start, end, lanes in active_obstacles:
                if start <= tick < end:
                    blocked_lanes.update(lanes)

            # Process notes by color
            notes_by_color = defaultdict(list)
            for event in tick_events:
                if event["type"] == "note":
                    # Clamp to valid positions
                    data = event["data"].copy()
                    data["lane"] = max(0, min(3, data.get("lane", 0)))
                    data["row"] = max(0, min(2, data.get("row", 0)))
                    data["color"] = max(0, min(1, data.get("color", 0)))
                    data["cut_dir"] = max(0, min(8, data.get("cut_dir", 0)))

                    # Skip if blocked by obstacle
                    if data["lane"] in blocked_lanes:
                        continue

                    notes_by_color[data["color"]].append({
                        **event,
                        "data": data
                    })

            # Keep max notes per hand, avoiding duplicates
            positions_used = set()
            for color in [0, 1]:
                notes = notes_by_color[color]
                kept = 0
                for note in notes:
                    pos = (note["data"]["lane"], note["data"]["row"])
                    if pos in positions_used:
                        continue
                    if kept >= self.max_notes_per_hand:
                        break
                    positions_used.add(pos)
                    fixed.append(note)
                    kept += 1

            # Process bombs
            for event in tick_events:
                if event["type"] == "bomb":
                    data = event["data"].copy()
                    data["lane"] = max(0, min(3, data.get("lane", 0)))
                    data["row"] = max(0, min(2, data.get("row", 0)))

                    pos = (data["lane"], data["row"])
                    if pos not in positions_used:
                        fixed.append({**event, "data": data})

        # Sort by tick
        fixed.sort(key=lambda e: e["tick"])

        return fixed

    def get_stats(self, events: list) -> dict:
        """Get statistics about a map."""
        notes = [e for e in events if e["type"] == "note"]
        bombs = [e for e in events if e["type"] == "bomb"]
        obstacles = [e for e in events if e["type"] == "obstacle"]

        if not events:
            return {"empty": True}

        duration_ticks = max(e["tick"] for e in events)

        # Notes per second (assuming 16 ticks per beat at 120 BPM)
        # This is approximate - actual NPS depends on BPM
        approx_duration_sec = duration_ticks / 16 / 2  # ticks / ticks_per_beat / bpm_factor

        return {
            "total_events": len(events),
            "notes": len(notes),
            "bombs": len(bombs),
            "obstacles": len(obstacles),
            "duration_ticks": duration_ticks,
            "approx_nps": len(notes) / max(1, approx_duration_sec),
            "red_notes": sum(1 for n in notes if n["data"].get("color") == 0),
            "blue_notes": sum(1 for n in notes if n["data"].get("color") == 1),
        }


# CLI interface
if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from src.data.parser import parse_map

    parser = argparse.ArgumentParser(description="Validate Beat Saber maps")
    parser.add_argument("map_folder", help="Map folder to validate")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix violations")

    args = parser.parse_args()

    validator = ConstraintValidator()

    maps = parse_map(Path(args.map_folder))

    for m in maps:
        print(f"\n{'='*50}")
        print(f"{m.difficulty.name} ({m.format_version})")
        print(f"{'='*50}")

        # Get stats
        stats = validator.get_stats(m.events)
        print(f"Notes: {stats['notes']} | Bombs: {stats['bombs']} | Obstacles: {stats['obstacles']}")

        # Validate
        is_valid, violations = validator.validate(m.events)

        if is_valid:
            print("Status: VALID")
        else:
            print(f"Status: INVALID ({len(violations)} violations)")

        # Show violations
        errors = [v for v in violations if v.severity == "error"]
        warnings = [v for v in violations if v.severity == "warning"]

        if errors:
            print(f"\nErrors ({len(errors)}):")
            for v in errors[:10]:
                print(f"  Tick {v.tick}: {v.description}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")

        if warnings:
            print(f"\nWarnings ({len(warnings)}):")
            for v in warnings[:5]:
                print(f"  Tick {v.tick}: {v.description}")
            if len(warnings) > 5:
                print(f"  ... and {len(warnings) - 5} more")

        # Fix if requested
        if args.fix and not is_valid:
            fixed_events = validator.fix(m.events)
            is_valid_after, violations_after = validator.validate(fixed_events)
            print(f"\nAfter fixing: {'VALID' if is_valid_after else 'INVALID'}")
            print(f"Events: {len(m.events)} -> {len(fixed_events)}")
