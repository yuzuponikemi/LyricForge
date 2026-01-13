"""
Video/Audio Downloader Module

Wrapper around yt-dlp for downloading videos and extracting audio from various sources.
Supports YouTube, Vimeo, and many other platforms.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yt_dlp

from core.context import ProcessingContext
from utils.logger import ContextualLogger


class DownloadError(Exception):
    """Exception raised when download fails."""

    pass


class Downloader:
    """
    Handles video/audio downloading using yt-dlp.
    """

    def __init__(self, context: ProcessingContext, logger: ContextualLogger):
        """
        Initialize downloader.

        Args:
            context: Processing context
            logger: Contextual logger
        """
        self.context = context
        self.logger = logger
        self.config = context.config.get("downloader", {})

    def download(self) -> Dict[str, Any]:
        """
        Download video and extract audio.

        Returns:
            Dictionary with download information and metadata

        Raises:
            DownloadError: If download fails
        """
        self.logger.info("Starting download", url=self.context.url)

        try:
            # First, extract metadata without downloading
            metadata = self._extract_metadata()

            # Set video ID in context
            video_id = metadata.get("id", "unknown")
            self.context.set_video_id(video_id)

            # Update context metadata
            self.context.update_metadata("title", metadata.get("title", "Unknown"))
            self.context.update_metadata("uploader", metadata.get("uploader", "Unknown"))
            self.context.update_metadata("duration", metadata.get("duration", 0))
            self.context.update_metadata(
                "upload_date", metadata.get("upload_date", "Unknown")
            )

            self.logger.info(
                "Metadata extracted",
                video_id=video_id,
                title=metadata.get("title", "Unknown"),
                duration=metadata.get("duration", 0),
            )

            # Download video
            video_path = self._download_video()
            self.logger.info("Video downloaded", path=str(video_path))

            # Download/extract audio
            audio_path = self._download_audio()
            self.logger.info("Audio extracted", path=str(audio_path))

            return {
                "video_path": video_path,
                "audio_path": audio_path,
                "metadata": metadata,
            }

        except Exception as e:
            error_msg = f"Download failed: {str(e)}"
            self.logger.error(error_msg)
            raise DownloadError(error_msg) from e

    def _extract_metadata(self) -> Dict[str, Any]:
        """
        Extract video metadata without downloading.

        Returns:
            Dictionary with metadata
        """
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(self.context.url, download=False)

        return info

    def _download_video(self) -> Path:
        """
        Download video file.

        Returns:
            Path to downloaded video

        Raises:
            DownloadError: If download fails
        """
        output_path = self.context.raw_video_path
        output_template = str(output_path.with_suffix(""))  # Remove extension

        ydl_opts = {
            "format": self.config.get(
                "format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
            ),
            "outtmpl": output_template,
            "quiet": False,
            "no_warnings": False,
            "writethumbnail": self.config.get("write_thumbnail", True),
            "writeinfojson": self.config.get("write_info_json", True),
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.context.url])
        except Exception as e:
            raise DownloadError(f"Video download failed: {str(e)}") from e

        # yt-dlp may add extension automatically, find the actual file
        video_file = self._find_downloaded_file(output_path)
        if video_file is None:
            raise DownloadError(f"Downloaded video file not found at {output_path}")

        return video_file

    def _download_audio(self) -> Path:
        """
        Extract audio from video or download audio directly.

        Returns:
            Path to audio file

        Raises:
            DownloadError: If extraction fails
        """
        output_path = self.context.raw_audio_path
        output_template = str(output_path.with_suffix(""))

        audio_format = self.config.get("audio_format", "wav")
        audio_quality = self.config.get("audio_quality", "0")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "quiet": False,
            "no_warnings": False,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": audio_format,
                    "preferredquality": audio_quality,
                }
            ],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.context.url])
        except Exception as e:
            raise DownloadError(f"Audio extraction failed: {str(e)}") from e

        # Find the actual audio file
        audio_file = self._find_downloaded_file(output_path)
        if audio_file is None:
            raise DownloadError(f"Extracted audio file not found at {output_path}")

        return audio_file

    def _find_downloaded_file(self, expected_path: Path) -> Optional[Path]:
        """
        Find the downloaded file (yt-dlp may add extensions).

        Args:
            expected_path: Expected file path

        Returns:
            Actual file path or None if not found
        """
        # Check exact path
        if expected_path.exists():
            return expected_path

        # Check common extensions
        extensions = [".mp4", ".mkv", ".webm", ".wav", ".m4a", ".mp3"]
        for ext in extensions:
            path = expected_path.with_suffix(ext)
            if path.exists():
                return path

        # Check parent directory for similar files
        if expected_path.parent.exists():
            stem = expected_path.stem
            for file in expected_path.parent.iterdir():
                if file.stem == stem:
                    return file

        return None


def create_downloader(context: ProcessingContext, logger: ContextualLogger) -> Downloader:
    """
    Factory function to create a Downloader instance.

    Args:
        context: Processing context
        logger: Contextual logger

    Returns:
        Downloader instance
    """
    return Downloader(context, logger)
