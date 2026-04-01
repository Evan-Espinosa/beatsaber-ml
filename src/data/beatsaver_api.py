"""
BeatSaver API Client

Handles downloading maps from BeatSaver with quality filtering.

API Reference: https://api.beatsaver.com/docs/
"""

import requests
import zipfile
import time
import json
import logging
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MapMetadata:
    """Metadata for a Beat Saber map."""
    id: str
    name: str
    hash: str
    bpm: float
    duration: int  # seconds
    upvotes: int
    downvotes: int
    score: float  # 0-1 rating
    download_url: str
    automapper: bool
    ranked: bool
    qualified: bool
    uploader: str
    difficulties: list

    @property
    def total_votes(self) -> int:
        return self.upvotes + self.downvotes

    @property
    def has_constant_bpm(self) -> bool:
        """Check if map has constant BPM (no BPM changes)."""
        # This is a heuristic - we'll need to check the actual map data
        # for _BPMChanges array during parsing
        return True  # Placeholder - actual check done during parsing


class BeatSaverAPI:
    """
    Client for the BeatSaver API.

    Handles:
    - Searching maps with quality filters (votes, rating, BPM)
    - Downloading map metadata
    - Downloading and extracting map files

    Example:
        api = BeatSaverAPI(output_dir='data/raw')

        # Search for high-quality maps
        for map_meta in api.search_maps(min_votes=50, min_ratio=0.75, limit=100):
            print(f"Found: {map_meta.name} ({map_meta.score:.2%} rating)")
            api.download_map(map_meta)
    """

    def __init__(
        self,
        api_base: str = "https://api.beatsaver.com",
        output_dir: str = "data/raw",
        rate_limit_delay: float = 0.5
    ):
        self.api_base = api_base.rstrip('/')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BeatSaberML/1.0 (https://github.com/beatsaber-ml)'
        })

    def _request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make a rate-limited request to the API."""
        url = f"{self.api_base}{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)  # Rate limiting
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                logger.warning(f"Resource not found: {url}")
                return {}
            elif response.status_code == 429:
                logger.warning("Rate limited, waiting 10 seconds...")
                time.sleep(10)
                return self._request(endpoint, params)
            else:
                logger.error(f"HTTP error {response.status_code}: {e}")
                raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def _parse_map_data(self, map_data: dict) -> Optional[MapMetadata]:
        """Parse API response into MapMetadata."""
        try:
            # Get the latest version
            versions = map_data.get('versions', [])
            if not versions:
                return None

            latest_version = versions[0]

            # Get difficulties from the version
            diffs = latest_version.get('diffs', [])
            difficulties = []
            for diff in diffs:
                difficulties.append({
                    'difficulty': diff.get('difficulty'),
                    'characteristic': diff.get('characteristic'),
                    'nps': diff.get('nps'),
                    'notes': diff.get('notes'),
                    'bombs': diff.get('bombs'),
                    'obstacles': diff.get('obstacles'),
                    'njs': diff.get('njs'),
                    'offset': diff.get('offset')
                })

            stats = map_data.get('stats', {})
            metadata = map_data.get('metadata', {})

            return MapMetadata(
                id=map_data.get('id'),
                name=map_data.get('name', 'Unknown'),
                hash=latest_version.get('hash'),
                bpm=metadata.get('bpm', 0),
                duration=metadata.get('duration', 0),
                upvotes=stats.get('upvotes', 0),
                downvotes=stats.get('downvotes', 0),
                score=stats.get('score', 0),
                download_url=latest_version.get('downloadURL'),
                automapper=map_data.get('automapper', False),
                ranked=map_data.get('ranked', False),
                qualified=map_data.get('qualified', False),
                uploader=map_data.get('uploader', {}).get('name', 'Unknown'),
                difficulties=difficulties
            )
        except (KeyError, IndexError) as e:
            logger.warning(f"Failed to parse map data: {e}")
            return None

    def get_map_metadata(self, map_id: str) -> Optional[MapMetadata]:
        """
        Get metadata for a specific map by ID.

        Args:
            map_id: The BeatSaver map ID (e.g., "1a2b3")

        Returns:
            MapMetadata or None if not found
        """
        data = self._request(f"/maps/id/{map_id}")
        if not data:
            return None
        return self._parse_map_data(data)

    def search_maps(
        self,
        min_votes: int = 50,
        min_ratio: float = 0.75,
        min_bpm: Optional[float] = None,
        max_bpm: Optional[float] = None,
        exclude_automapper: bool = True,
        limit: int = 1000,
        page_size: int = 20
    ) -> Generator[MapMetadata, None, None]:
        """
        Search for maps matching quality criteria.

        Args:
            min_votes: Minimum total votes (upvotes + downvotes)
            min_ratio: Minimum upvote ratio (0-1, e.g., 0.75 = 75%)
            min_bpm: Minimum BPM filter
            max_bpm: Maximum BPM filter
            exclude_automapper: If True, exclude auto-generated maps
            limit: Maximum number of maps to return
            page_size: Results per page (max 20)

        Yields:
            MapMetadata objects matching criteria
        """
        page = 0
        yielded = 0

        while yielded < limit:
            params = {
                'minRating': min_ratio,
                'sortOrder': 'Rating',  # Sort by rating (highest first)
                'pageSize': min(page_size, 20)
            }

            # API default excludes automapper, so only set if we WANT automapper
            if not exclude_automapper:
                params['automapper'] = 'false'  # Inverted: false = include automapper only

            if min_bpm is not None:
                params['minBpm'] = min_bpm
            if max_bpm is not None:
                params['maxBpm'] = max_bpm

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            data = self._request(f"/search/text/{page}", params)

            if not data:
                break

            docs = data.get('docs', [])
            if not docs:
                break

            for map_data in docs:
                if yielded >= limit:
                    break

                map_meta = self._parse_map_data(map_data)
                if map_meta is None:
                    continue

                # Additional filtering
                if map_meta.total_votes < min_votes:
                    continue
                if map_meta.score < min_ratio:
                    continue
                if exclude_automapper and map_meta.automapper:
                    continue

                yielded += 1
                yield map_meta

            page += 1

            # Check if we've reached the end
            if len(docs) < page_size:
                break

        logger.info(f"Search complete: {yielded} maps found")

    def download_map(
        self,
        map_meta: MapMetadata,
        extract: bool = True
    ) -> Optional[Path]:
        """
        Download a map and optionally extract it.

        Args:
            map_meta: MapMetadata object with download URL
            extract: If True, extract the zip file

        Returns:
            Path to the extracted folder or zip file, or None on failure
        """
        if not map_meta.download_url:
            logger.error(f"No download URL for map {map_meta.id}")
            return None

        # Create output directory for this map
        map_dir = self.output_dir / map_meta.id

        # Skip if already downloaded
        if map_dir.exists() and any(map_dir.iterdir()):
            logger.info(f"Map {map_meta.id} already downloaded, skipping")
            return map_dir

        try:
            logger.info(f"Downloading map: {map_meta.name} ({map_meta.id})")

            # Download the zip file
            response = self.session.get(map_meta.download_url, timeout=60)
            response.raise_for_status()

            time.sleep(self.rate_limit_delay)

            if extract:
                # Extract directly to map directory
                map_dir.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(BytesIO(response.content)) as zf:
                    zf.extractall(map_dir)

                # Save metadata
                metadata_path = map_dir / 'beatsaver_metadata.json'
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'id': map_meta.id,
                        'name': map_meta.name,
                        'hash': map_meta.hash,
                        'bpm': map_meta.bpm,
                        'duration': map_meta.duration,
                        'upvotes': map_meta.upvotes,
                        'downvotes': map_meta.downvotes,
                        'score': map_meta.score,
                        'automapper': map_meta.automapper,
                        'ranked': map_meta.ranked,
                        'uploader': map_meta.uploader,
                        'difficulties': map_meta.difficulties
                    }, f, indent=2)

                logger.info(f"Extracted to: {map_dir}")
                return map_dir
            else:
                # Save as zip
                zip_path = self.output_dir / f"{map_meta.id}.zip"
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Saved to: {zip_path}")
                return zip_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed for {map_meta.id}: {e}")
            return None
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file for {map_meta.id}: {e}")
            return None

    def download_maps(
        self,
        min_votes: int = 50,
        min_ratio: float = 0.75,
        limit: int = 1000,
        **search_kwargs
    ) -> list[Path]:
        """
        Search and download maps matching criteria.

        Convenience method combining search_maps() and download_map().

        Returns:
            List of paths to downloaded map folders
        """
        downloaded = []

        for map_meta in self.search_maps(
            min_votes=min_votes,
            min_ratio=min_ratio,
            limit=limit,
            **search_kwargs
        ):
            path = self.download_map(map_meta)
            if path:
                downloaded.append(path)

        logger.info(f"Downloaded {len(downloaded)} maps")
        return downloaded


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download Beat Saber maps from BeatSaver')
    parser.add_argument('--output', '-o', default='data/raw', help='Output directory')
    parser.add_argument('--min-votes', type=int, default=50, help='Minimum votes')
    parser.add_argument('--min-ratio', type=float, default=0.75, help='Minimum upvote ratio')
    parser.add_argument('--limit', type=int, default=100, help='Maximum maps to download')
    parser.add_argument('--list-only', action='store_true', help='List maps without downloading')

    args = parser.parse_args()

    api = BeatSaverAPI(output_dir=args.output)

    if args.list_only:
        print(f"\nSearching for maps (votes >= {args.min_votes}, ratio >= {args.min_ratio:.0%})...\n")
        for i, map_meta in enumerate(api.search_maps(
            min_votes=args.min_votes,
            min_ratio=args.min_ratio,
            limit=args.limit
        )):
            print(f"{i+1}. [{map_meta.id}] {map_meta.name}")
            print(f"   BPM: {map_meta.bpm} | Votes: {map_meta.total_votes} | Rating: {map_meta.score:.1%}")
            print(f"   Uploader: {map_meta.uploader}")
            print()
    else:
        print(f"\nDownloading maps to {args.output}...")
        paths = api.download_maps(
            min_votes=args.min_votes,
            min_ratio=args.min_ratio,
            limit=args.limit
        )
        print(f"\nDownloaded {len(paths)} maps")
