"""
Execution Context Management for LyricForge

This module manages the state and paths throughout the pipeline execution.
Each processing run gets its own context with unique identifiers and file paths.
"""

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ProcessingContext:
    """
    Holds all state and metadata for a single pipeline execution.

    This class encapsulates:
    - Input parameters (URL, video ID)
    - File paths for all stages (raw, stems, subtitles, output)
    - Metadata extracted during processing
    - Timestamps for tracking progress
    """

    # Input information
    url: str
    video_id: Optional[str] = None

    # Unique execution identifier
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Timestamps
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # File paths (populated during execution)
    raw_video_path: Optional[Path] = None
    raw_audio_path: Optional[Path] = None
    vocal_stem_path: Optional[Path] = None
    instrumental_stem_path: Optional[Path] = None
    raw_transcript_path: Optional[Path] = None
    refined_transcript_path: Optional[Path] = None
    subtitle_path: Optional[Path] = None
    output_video_path: Optional[Path] = None
    lyrics_file_path: Optional[Path] = None

    # Lyrics data (for forced alignment)
    lyrics_text: Optional[str] = None
    artist: Optional[str] = None
    song_title: Optional[str] = None

    # Metadata (extracted during processing)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Processing status
    status: str = "initialized"  # initialized, downloading, separating, transcribing, refining, compositing, completed, failed
    error: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize paths based on configuration."""
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        paths_to_create = [
            self.get_path("raw"),
            self.get_path("stems"),
            self.get_path("subtitles"),
            self.get_path("output"),
        ]

        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)

    def get_path(self, directory: str) -> Path:
        """
        Get the Path object for a specific directory.

        Args:
            directory: One of 'raw', 'stems', 'subtitles', 'output'

        Returns:
            Path object for the requested directory
        """
        path_mapping = {
            "raw": self.config.get("paths", {}).get("raw", "data/raw"),
            "stems": self.config.get("paths", {}).get("stems", "data/stems"),
            "subtitles": self.config.get("paths", {}).get("subtitles", "data/subs"),
            "output": self.config.get("paths", {}).get("output", "data/output"),
        }

        return Path(path_mapping.get(directory, f"data/{directory}"))

    def set_video_id(self, video_id: str) -> None:
        """Set video ID and initialize file paths based on it."""
        self.video_id = video_id
        base_name = f"{video_id}_{self.run_id}"

        # Set default paths
        self.raw_video_path = self.get_path("raw") / f"{base_name}.mp4"
        self.raw_audio_path = self.get_path("raw") / f"{base_name}.wav"
        self.vocal_stem_path = self.get_path("stems") / f"{base_name}_vocals.wav"
        self.instrumental_stem_path = self.get_path("stems") / f"{base_name}_instrumental.wav"
        self.raw_transcript_path = self.get_path("subtitles") / f"{base_name}_raw.json"
        self.refined_transcript_path = self.get_path("subtitles") / f"{base_name}_refined.json"
        self.subtitle_path = self.get_path("subtitles") / f"{base_name}.srt"
        self.output_video_path = self.get_path("output") / f"{base_name}_subtitled.mp4"

    def update_metadata(self, key: str, value: Any) -> None:
        """Add or update metadata."""
        self.metadata[key] = value

    def mark_completed(self) -> None:
        """Mark the processing as completed."""
        self.status = "completed"
        self.end_time = datetime.now()

    def mark_failed(self, error_message: str) -> None:
        """Mark the processing as failed."""
        self.status = "failed"
        self.error = error_message
        self.end_time = datetime.now()

    def get_duration(self) -> Optional[float]:
        """Get the total processing duration in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "url": self.url,
            "video_id": self.video_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.get_duration(),
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata,
            "paths": {
                "raw_video": str(self.raw_video_path) if self.raw_video_path else None,
                "raw_audio": str(self.raw_audio_path) if self.raw_audio_path else None,
                "vocal_stem": str(self.vocal_stem_path) if self.vocal_stem_path else None,
                "subtitle": str(self.subtitle_path) if self.subtitle_path else None,
                "output_video": str(self.output_video_path) if self.output_video_path else None,
            },
        }


class ConfigLoader:
    """Utility class for loading configuration files."""

    @staticmethod
    def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dictionary containing configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config or {}

    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.

        Args:
            base_config: Base configuration
            override_config: Configuration to override base with

        Returns:
            Merged configuration dictionary
        """
        result = base_config.copy()

        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader.merge_configs(result[key], value)
            else:
                result[key] = value

        return result


def create_context(
    url: str,
    config_path: str = "config/settings.yaml",
    config_overrides: Optional[Dict[str, Any]] = None,
) -> ProcessingContext:
    """
    Factory function to create a new ProcessingContext.

    Args:
        url: Video URL to process
        config_path: Path to configuration file
        config_overrides: Optional configuration overrides

    Returns:
        Initialized ProcessingContext
    """
    config = ConfigLoader.load_config(config_path)

    if config_overrides:
        config = ConfigLoader.merge_configs(config, config_overrides)

    context = ProcessingContext(url=url, config=config)

    return context
