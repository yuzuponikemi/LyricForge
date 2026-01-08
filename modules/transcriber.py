"""
Audio Transcription Module

Wrapper around Faster-Whisper for speech-to-text transcription.
Includes support for VAD (Voice Activity Detection) and word-level timestamps.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from faster_whisper import WhisperModel

from core.context import ProcessingContext
from utils.gpu_manager import GPUManager, GPUMemoryContext
from utils.logger import ContextualLogger


class TranscriptionError(Exception):
    """Exception raised when transcription fails."""

    pass


class Transcriber:
    """
    Handles audio transcription using Faster-Whisper.
    """

    def __init__(
        self,
        context: ProcessingContext,
        logger: ContextualLogger,
        gpu_manager: GPUManager,
    ):
        """
        Initialize transcriber.

        Args:
            context: Processing context
            logger: Contextual logger
            gpu_manager: GPU manager for device selection
        """
        self.context = context
        self.logger = logger
        self.gpu_manager = gpu_manager
        self.config = context.config.get("transcriber", {})
        self.model: Optional[WhisperModel] = None

    def transcribe(self, audio_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file (uses context if not provided)

        Returns:
            Dictionary with transcription segments

        Raises:
            TranscriptionError: If transcription fails
        """
        if audio_path is None:
            # Try to use vocal stem if available, otherwise use raw audio
            if self.context.vocal_stem_path and self.context.vocal_stem_path.exists():
                audio_path = self.context.vocal_stem_path
            else:
                audio_path = self.context.raw_audio_path

        if audio_path is None or not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        self.logger.info("Starting transcription", audio_file=str(audio_path))

        try:
            with GPUMemoryContext(self.gpu_manager):
                # Load model
                self._load_model()

                # Transcribe
                segments = self._transcribe_audio(audio_path)

                # Save raw transcription
                self._save_transcription(segments, self.context.raw_transcript_path)

                self.logger.info(
                    "Transcription completed",
                    segment_count=len(segments),
                    output=str(self.context.raw_transcript_path),
                )

                return {"segments": segments, "path": self.context.raw_transcript_path}

        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            self.logger.error(error_msg)
            raise TranscriptionError(error_msg) from e
        finally:
            self._unload_model()

    def _load_model(self) -> None:
        """Load Whisper model."""
        model_name = self.config.get("model", "large-v3")
        device = self.gpu_manager.get_optimal_device()
        compute_type = self.config.get(
            "compute_type", self.gpu_manager.get_compute_type(device)
        )

        self.logger.info(
            "Loading Whisper model",
            model=model_name,
            device=device,
            compute_type=compute_type,
        )

        try:
            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=None,  # Use default cache
            )
        except Exception as e:
            raise TranscriptionError(f"Failed to load Whisper model: {str(e)}") from e

    def _unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.gpu_manager.clear_gpu_memory()

    def _transcribe_audio(self, audio_path: Path) -> List[Dict[str, Any]]:
        """
        Transcribe audio file and return segments.

        Args:
            audio_path: Path to audio file

        Returns:
            List of segment dictionaries
        """
        if self.model is None:
            raise TranscriptionError("Model not loaded")

        # Get transcription parameters
        language = self.config.get("language")
        beam_size = self.config.get("beam_size", 5)
        vad_filter = self.config.get("vad_filter", True)
        vad_parameters = self.config.get("vad_parameters", {})

        self.logger.info(
            "Transcribing audio",
            language=language or "auto",
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        # Transcribe
        segments_generator, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters if vad_filter else None,
            word_timestamps=True,  # Enable word-level timestamps
        )

        # Convert generator to list and extract relevant information
        segments = []
        for segment in segments_generator:
            segment_dict = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            }

            # Add word-level timestamps if available
            if hasattr(segment, "words") and segment.words:
                segment_dict["words"] = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                    for word in segment.words
                ]

            segments.append(segment_dict)

        self.logger.info(
            "Audio transcribed",
            detected_language=info.language,
            language_probability=f"{info.language_probability:.2f}",
            segments=len(segments),
        )

        return segments

    def _save_transcription(
        self, segments: List[Dict[str, Any]], output_path: Optional[Path]
    ) -> None:
        """
        Save transcription to JSON file.

        Args:
            segments: List of segment dictionaries
            output_path: Output file path
        """
        if output_path is None:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        transcription_data = {
            "segments": segments,
            "metadata": {
                "model": self.config.get("model", "large-v3"),
                "language": self.config.get("language", "auto"),
                "segment_count": len(segments),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcription_data, f, ensure_ascii=False, indent=2)

        self.logger.debug("Transcription saved", path=str(output_path))


def create_transcriber(
    context: ProcessingContext,
    logger: ContextualLogger,
    gpu_manager: GPUManager,
) -> Transcriber:
    """
    Factory function to create a Transcriber instance.

    Args:
        context: Processing context
        logger: Contextual logger
        gpu_manager: GPU manager

    Returns:
        Transcriber instance
    """
    return Transcriber(context, logger, gpu_manager)
