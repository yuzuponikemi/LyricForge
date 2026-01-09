"""
Pipeline Manager for LyricForge

Orchestrates the entire processing flow from download to final output.
Implements the Pipeline Pattern with clear separation of concerns.
"""

from typing import Any, Dict, Optional

from core.context import ProcessingContext, create_context
from modules.aligner import Aligner, create_aligner
from modules.calibrator import Calibrator, create_calibrator
from modules.downloader import Downloader, create_downloader
from modules.lyrics_fetcher import LyricsFetcher, create_lyrics_fetcher
from modules.refiner import Refiner, create_refiner
from modules.separator import Separator, create_separator
from modules.transcriber import Transcriber, create_transcriber
from utils.gpu_manager import GPUManager, create_gpu_manager
from utils.logger import ContextualLogger, LyricForgeLogger, get_contextual_logger, setup_logger


class PipelineError(Exception):
    """Exception raised when pipeline execution fails."""

    pass


class PipelineManager:
    """
    Manages the execution of the entire processing pipeline.

    Pipeline stages:
    1. Download (video/audio)
    2. Separation (optional - Demucs) ✅ Milestone 2
    3. Transcription (Whisper)
    4. Refinement (optional - LLM) ✅ Milestone 2
    5. Subtitle Generation (SRT/ASS, will be implemented in Milestone 3)
    6. Video Composition (optional - FFmpeg, will be implemented in Milestone 3)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger_base = setup_logger(config)
        self.gpu_manager = create_gpu_manager(config)

    def process(
        self,
        url: str,
        stages: Optional[list[str]] = None,
    ) -> ProcessingContext:
        """
        Execute the processing pipeline.

        Args:
            url: Video URL to process
            stages: List of stages to execute (default: all available stages)

        Returns:
            ProcessingContext with results

        Raises:
            PipelineError: If pipeline execution fails
        """
        # Create context
        context = create_context(url, config_overrides={"paths": self.config.get("paths", {})})
        context.config = self.config

        # Create contextual logger
        logger = get_contextual_logger(self.logger_base, context.run_id)

        logger.info("Pipeline execution started", url=url)

        # Print GPU info
        self._log_system_info(logger)

        # Default stages for Milestone 2 (full pipeline with separation and refinement)
        if stages is None:
            stages = ["download", "separate", "transcribe", "refine"]

        try:
            # Execute stages
            for stage in stages:
                self._execute_stage(stage, context, logger)

            # Mark as completed
            context.mark_completed()
            logger.info(
                "Pipeline execution completed",
                duration=f"{context.get_duration():.2f}s",
            )

            return context

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            context.mark_failed(error_msg)
            logger.error(error_msg)
            raise PipelineError(error_msg) from e

    def _execute_stage(
        self,
        stage: str,
        context: ProcessingContext,
        logger: ContextualLogger,
    ) -> None:
        """
        Execute a single pipeline stage.

        Args:
            stage: Stage name
            context: Processing context
            logger: Contextual logger

        Raises:
            PipelineError: If stage execution fails
        """
        logger.info(f"Starting stage: {stage}")
        context.status = stage

        if stage == "download":
            self._stage_download(context, logger)
        elif stage == "calibrate":
            self._stage_calibrate(context, logger)
        elif stage == "separate":
            self._stage_separate(context, logger)
        elif stage == "transcribe":
            self._stage_transcribe(context, logger)
        elif stage == "align":
            self._stage_align(context, logger)
        elif stage == "refine":
            self._stage_refine(context, logger)
        elif stage == "subtitle":
            # Will be implemented in Milestone 3
            logger.warning("Subtitle generation not yet implemented (Milestone 3)")
        elif stage == "compose":
            # Will be implemented in Milestone 3
            logger.warning("Video composition not yet implemented (Milestone 3)")
        else:
            raise PipelineError(f"Unknown stage: {stage}")

        logger.info(f"Completed stage: {stage}")

    def _stage_download(
        self,
        context: ProcessingContext,
        logger: ContextualLogger,
    ) -> None:
        """
        Execute download stage.

        Args:
            context: Processing context
            logger: Contextual logger
        """
        downloader = create_downloader(context, logger)
        result = downloader.download()

        # Update context paths (may have been modified by downloader)
        context.raw_video_path = result["video_path"]
        context.raw_audio_path = result["audio_path"]

    def _stage_separate(
        self,
        context: ProcessingContext,
        logger: ContextualLogger,
    ) -> None:
        """
        Execute separation stage.

        Args:
            context: Processing context
            logger: Contextual logger
        """
        separator = create_separator(context, logger, self.gpu_manager)
        result = separator.separate()

        # Context is already updated by separator
        logger.debug("Separation result saved", vocal_path=str(result.get("vocals")))

    def _stage_transcribe(
        self,
        context: ProcessingContext,
        logger: ContextualLogger,
    ) -> None:
        """
        Execute transcription stage.

        Args:
            context: Processing context
            logger: Contextual logger
        """
        transcriber = create_transcriber(context, logger, self.gpu_manager)
        result = transcriber.transcribe()

        # Context is already updated by transcriber
        logger.debug("Transcription result saved", path=str(result["path"]))

    def _stage_refine(
        self,
        context: ProcessingContext,
        logger: ContextualLogger,
    ) -> None:
        """
        Execute refinement stage.

        Args:
            context: Processing context
            logger: Contextual logger
        """
        refiner = create_refiner(context, logger)
        result = refiner.refine()

        # Context is already updated by refiner
        if result:
            logger.debug("Refinement result saved", path=str(result.get("path")))

    def _stage_calibrate(
        self,
        context: ProcessingContext,
        logger: ContextualLogger,
    ) -> None:
        """
        Execute calibration stage to find optimal parameters.

        Args:
            context: Processing context
            logger: Contextual logger
        """
        calibrator = create_calibrator(context, logger, self.gpu_manager)

        # Get sample duration from config
        sample_duration = context.config.get("calibrator", {}).get("sample_duration", 30)

        # Run calibration
        result = calibrator.calibrate(sample_duration=sample_duration)

        # Apply optimal parameters to context config
        optimal_params = result.get("parameters", {})

        if "separator" in optimal_params:
            context.config["separator"].update(optimal_params["separator"])
            logger.info("Applied optimal separator parameters", params=optimal_params["separator"])

        if "transcriber" in optimal_params:
            context.config["transcriber"].update(optimal_params["transcriber"])
            logger.info("Applied optimal transcriber parameters", params=optimal_params["transcriber"])

        # Store calibration result in metadata
        context.update_metadata("calibration_preset", result.get("name"))
        context.update_metadata("calibration_score", result.get("score"))

    def _stage_align(
        self,
        context: ProcessingContext,
        logger: ContextualLogger,
    ) -> None:
        """
        Execute forced alignment stage with known lyrics.

        Args:
            context: Processing context
            logger: Contextual logger
        """
        # Get lyrics information from config
        lyrics_config = context.config.get("lyrics", {})

        lyrics_file_str = lyrics_config.get("file")
        artist = lyrics_config.get("artist")
        song_title = lyrics_config.get("song_title")

        # Convert to Path if string
        from pathlib import Path as PathLib
        lyrics_file = PathLib(lyrics_file_str) if lyrics_file_str else None

        # Set in context
        if lyrics_file:
            context.lyrics_file_path = lyrics_file
        if artist:
            context.artist = artist
        if song_title:
            context.song_title = song_title

        # Fetch lyrics if not already available
        if not context.lyrics_text:
            fetcher = create_lyrics_fetcher(context, logger)

            try:
                lyrics_text = fetcher.fetch(
                    lyrics_file=context.lyrics_file_path,
                    artist=context.artist,
                    song_title=context.song_title,
                )
                context.lyrics_text = lyrics_text
            except Exception as e:
                logger.error(f"Failed to fetch lyrics: {e}")
                raise PipelineError(
                    "Forced alignment requires lyrics. "
                    "Use --lyrics-file or --artist + --song-title"
                ) from e

        # Run forced alignment
        aligner = create_aligner(context, logger, self.gpu_manager)
        result = aligner.align(context.lyrics_text)

        # Context is already updated by aligner
        logger.debug("Forced alignment result saved", path=str(result.get("path")))

    def _log_system_info(self, logger: ContextualLogger) -> None:
        """
        Log system and GPU information.

        Args:
            logger: Contextual logger
        """
        device_info = self.gpu_manager.get_device_info()
        optimal_device = self.gpu_manager.get_optimal_device()

        logger.info(
            "System information",
            available_devices=device_info["available_devices"],
            optimal_device=optimal_device,
        )

        if device_info["available_devices"]["cuda"]:
            logger.info(
                "CUDA device detected",
                device_count=device_info.get("cuda_device_count", 0),
                device_name=device_info.get("cuda_device_name", "Unknown"),
                total_memory_gb=f"{device_info.get('cuda_memory_total', 0) / (1024**3):.2f}",
            )


def create_pipeline(
    config_path: str = "config/settings.yaml",
    config_overrides: Optional[Dict[str, Any]] = None,
) -> PipelineManager:
    """
    Factory function to create a pipeline manager.

    Args:
        config_path: Path to configuration file
        config_overrides: Optional configuration overrides

    Returns:
        PipelineManager instance
    """
    from core.context import ConfigLoader

    config = ConfigLoader.load_config(config_path)

    if config_overrides:
        config = ConfigLoader.merge_configs(config, config_overrides)

    return PipelineManager(config)


def run_pipeline(
    url: str,
    config_path: str = "config/settings.yaml",
    stages: Optional[list[str]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    lyrics_file: Optional[Path] = None,
    artist: Optional[str] = None,
    song_title: Optional[str] = None,
) -> ProcessingContext:
    """
    Convenience function to run the pipeline.

    Args:
        url: Video URL to process
        config_path: Path to configuration file
        stages: List of stages to execute
        config_overrides: Optional configuration overrides
        lyrics_file: Optional path to lyrics file for forced alignment
        artist: Optional artist name for lyrics search
        song_title: Optional song title for lyrics search

    Returns:
        ProcessingContext with results
    """
    # Pass lyrics information via config_overrides
    if lyrics_file or artist or song_title:
        if config_overrides is None:
            config_overrides = {}
        config_overrides.setdefault("lyrics", {})
        if lyrics_file:
            config_overrides["lyrics"]["file"] = str(lyrics_file)
        if artist:
            config_overrides["lyrics"]["artist"] = artist
        if song_title:
            config_overrides["lyrics"]["song_title"] = song_title

    pipeline = create_pipeline(config_path, config_overrides)
    return pipeline.process(url, stages)
