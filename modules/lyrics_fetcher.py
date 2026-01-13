"""
Lyrics Fetching Module

Retrieves lyrics from various sources for forced alignment.
Supports manual files, Genius API, and other providers.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from core.context import ProcessingContext
from utils.logger import ContextualLogger


class LyricsFetchError(Exception):
    """Exception raised when lyrics fetching fails."""

    pass


class LyricsFetcher:
    """
    Handles lyrics retrieval from various sources.
    """

    def __init__(
        self,
        context: ProcessingContext,
        logger: ContextualLogger,
    ):
        """
        Initialize lyrics fetcher.

        Args:
            context: Processing context
            logger: Contextual logger
        """
        self.context = context
        self.logger = logger
        self.config = context.config.get("lyrics_fetcher", {})

    def fetch(
        self,
        lyrics_file: Optional[Path] = None,
        artist: Optional[str] = None,
        song_title: Optional[str] = None,
    ) -> str:
        """
        Fetch lyrics from the best available source.

        Args:
            lyrics_file: Path to manual lyrics file
            artist: Artist name for API search
            song_title: Song title for API search

        Returns:
            Lyrics text

        Raises:
            LyricsFetchError: If lyrics cannot be fetched
        """
        self.logger.info(
            "Fetching lyrics",
            file=str(lyrics_file) if lyrics_file else None,
            artist=artist,
            song_title=song_title,
        )

        # Priority 1: Manual file
        if lyrics_file and lyrics_file.exists():
            lyrics = self._load_from_file(lyrics_file)
            self.logger.info("Lyrics loaded from file", path=str(lyrics_file))
            return lyrics

        # Priority 2: API search
        if artist and song_title:
            try:
                lyrics = self._fetch_from_api(artist, song_title)
                self.logger.info("Lyrics fetched from API", artist=artist, title=song_title)
                return lyrics
            except Exception as e:
                self.logger.warning(f"API fetch failed: {e}")

        # If all methods fail
        raise LyricsFetchError(
            "Could not fetch lyrics. Provide --lyrics-file or --artist + --song-title"
        )

    def _load_from_file(self, lyrics_file: Path) -> str:
        """
        Load lyrics from a text file.

        Args:
            lyrics_file: Path to lyrics file

        Returns:
            Lyrics text

        Raises:
            LyricsFetchError: If file cannot be read
        """
        try:
            with open(lyrics_file, "r", encoding="utf-8") as f:
                lyrics = f.read()

            # Clean up lyrics
            lyrics = self._clean_lyrics(lyrics)

            if not lyrics.strip():
                raise LyricsFetchError("Lyrics file is empty")

            return lyrics

        except Exception as e:
            raise LyricsFetchError(f"Failed to read lyrics file: {str(e)}") from e

    def _fetch_from_api(self, artist: str, song_title: str) -> str:
        """
        Fetch lyrics from API (Genius).

        Args:
            artist: Artist name
            song_title: Song title

        Returns:
            Lyrics text

        Raises:
            LyricsFetchError: If API fetch fails
        """
        provider = self.config.get("provider", "genius")

        if provider == "genius":
            return self._fetch_from_genius(artist, song_title)
        else:
            raise LyricsFetchError(f"Unknown provider: {provider}")

    def _fetch_from_genius(self, artist: str, song_title: str) -> str:
        """
        Fetch lyrics from Genius API.

        Args:
            artist: Artist name
            song_title: Song title

        Returns:
            Lyrics text

        Raises:
            LyricsFetchError: If fetch fails
        """
        api_key = self.config.get("genius_api_key")

        if not api_key:
            raise LyricsFetchError(
                "Genius API key not configured. Set 'genius_api_key' in config or use --lyrics-file"
            )

        try:
            # Search for song
            search_url = "https://api.genius.com/search"
            headers = {"Authorization": f"Bearer {api_key}"}
            params = {"q": f"{artist} {song_title}"}

            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            hits = data.get("response", {}).get("hits", [])

            if not hits:
                raise LyricsFetchError(f"No results found for '{artist} - {song_title}'")

            # Get first result
            song_url = hits[0]["result"]["url"]

            # Note: Genius API doesn't provide lyrics directly
            # Would need to scrape the page or use a library like lyricsgenius
            raise LyricsFetchError(
                "Genius API integration requires scraping. Please use --lyrics-file for now."
            )

        except requests.exceptions.RequestException as e:
            raise LyricsFetchError(f"Genius API request failed: {str(e)}") from e

    def _clean_lyrics(self, lyrics: str) -> str:
        """
        Clean and normalize lyrics text.

        Args:
            lyrics: Raw lyrics text

        Returns:
            Cleaned lyrics text
        """
        # Remove common metadata patterns
        lyrics = re.sub(r"\[.*?\]", "", lyrics)  # Remove [Verse], [Chorus], etc.
        lyrics = re.sub(r"\(.*?\)", "", lyrics)  # Remove (background vocals), etc.

        # Normalize whitespace
        lyrics = re.sub(r"\n\s*\n", "\n\n", lyrics)  # Double newlines for verses
        lyrics = lyrics.strip()

        return lyrics

    def save_lyrics(self, lyrics: str, output_path: Optional[Path] = None) -> Path:
        """
        Save fetched lyrics to a file.

        Args:
            lyrics: Lyrics text
            output_path: Output file path (optional)

        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = self.context.get_path("subtitles") / f"{self.context.video_id}_lyrics.txt"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(lyrics)

        self.logger.debug("Lyrics saved", path=str(output_path))

        return output_path


def create_lyrics_fetcher(
    context: ProcessingContext,
    logger: ContextualLogger,
) -> LyricsFetcher:
    """
    Factory function to create a LyricsFetcher instance.

    Args:
        context: Processing context
        logger: Contextual logger

    Returns:
        LyricsFetcher instance
    """
    return LyricsFetcher(context, logger)
