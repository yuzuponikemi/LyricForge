"""
Audio Source Separation Module

Wrapper around Demucs for separating audio into stems (vocals, drums, bass, other).
Optimized for lyric extraction by isolating vocal tracks.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from core.context import ProcessingContext
from utils.gpu_manager import GPUManager, GPUMemoryContext
from utils.logger import ContextualLogger


class SeparationError(Exception):
    """Exception raised when audio separation fails."""

    pass


class Separator:
    """
    Handles audio source separation using Demucs.
    """

    def __init__(
        self,
        context: ProcessingContext,
        logger: ContextualLogger,
        gpu_manager: GPUManager,
    ):
        """
        Initialize separator.

        Args:
            context: Processing context
            logger: Contextual logger
            gpu_manager: GPU manager for device selection
        """
        self.context = context
        self.logger = logger
        self.gpu_manager = gpu_manager
        self.config = context.config.get("separator", {})

    def separate(self, audio_path: Optional[Path] = None) -> Dict[str, Path]:
        """
        Separate audio into stems.

        Args:
            audio_path: Path to audio file (uses context if not provided)

        Returns:
            Dictionary with paths to separated stems

        Raises:
            SeparationError: If separation fails
        """
        if not self.config.get("enabled", True):
            self.logger.warning("Separator is disabled in config, skipping")
            return {}

        if audio_path is None:
            audio_path = self.context.raw_audio_path

        if audio_path is None or not audio_path.exists():
            raise SeparationError(f"Audio file not found: {audio_path}")

        self.logger.info("Starting audio separation", audio_file=str(audio_path))

        try:
            with GPUMemoryContext(self.gpu_manager):
                stems = self._separate_audio(audio_path)

                self.logger.info(
                    "Audio separation completed",
                    vocal_track=str(stems.get("vocals")),
                )

                return stems

        except Exception as e:
            error_msg = f"Audio separation failed: {str(e)}"
            self.logger.error(error_msg)
            raise SeparationError(error_msg) from e

    def _separate_audio(self, audio_path: Path) -> Dict[str, Path]:
        """
        Separate audio using Demucs.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with stem paths
        """
        model = self.config.get("model", "htdemucs")
        device = self._get_device()
        shifts = self.config.get("shifts", 1)
        split = self.config.get("split", True)
        overlap = self.config.get("overlap", 0.25)
        jobs = self.config.get("jobs", 0)

        # Output directory
        output_dir = self.context.get_path("stems")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Running Demucs",
            model=model,
            device=device,
            shifts=shifts,
            split=split,
        )

        # Build demucs command
        cmd = [
            "demucs",
            "--two-stems=vocals",  # Only separate vocals (faster)
            f"--name={model}",
            f"--device={device}",
            f"--shifts={shifts}",
            f"--overlap={overlap}",
            f"--out={output_dir}",
        ]

        if split:
            cmd.append("--split")

        if jobs > 0:
            cmd.append(f"--jobs={jobs}")

        cmd.append(str(audio_path))

        # Run demucs
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            self.logger.debug("Demucs stdout", output=result.stdout[-500:] if result.stdout else "")

        except subprocess.CalledProcessError as e:
            self.logger.error("Demucs failed", stderr=e.stderr[-500:] if e.stderr else "")
            raise SeparationError(f"Demucs command failed: {e.stderr}") from e
        except FileNotFoundError:
            raise SeparationError(
                "Demucs not found. Install it with: pip install demucs"
            ) from None

        # Find separated files
        stems = self._find_separated_stems(audio_path, output_dir, model)

        # Move to final location and update context
        self._organize_stems(stems)

        return {
            "vocals": self.context.vocal_stem_path,
            "instrumental": self.context.instrumental_stem_path,
        }

    def _get_device(self) -> str:
        """
        Get device for Demucs.

        Returns:
            Device string for Demucs ('cuda', 'cpu')
        """
        config_device = self.config.get("device", "auto")

        if config_device == "auto":
            optimal = self.gpu_manager.get_optimal_device()
            # Demucs uses different naming
            if optimal == "cuda":
                return "cuda"
            elif optimal == "mps":
                # Demucs doesn't support MPS directly, fall back to CPU
                self.logger.warning("Demucs doesn't support MPS, using CPU")
                return "cpu"
            else:
                return "cpu"
        else:
            return config_device

    def _find_separated_stems(
        self, audio_path: Path, output_dir: Path, model: str
    ) -> Dict[str, Path]:
        """
        Find separated stem files in Demucs output directory.

        Args:
            audio_path: Original audio file path
            output_dir: Demucs output directory
            model: Model name used

        Returns:
            Dictionary with stem paths
        """
        # Demucs creates: output_dir/model_name/audio_stem/vocals.wav
        audio_stem = audio_path.stem
        model_output = output_dir / model / audio_stem

        if not model_output.exists():
            raise SeparationError(f"Demucs output directory not found: {model_output}")

        vocals_path = model_output / "vocals.wav"
        no_vocals_path = model_output / "no_vocals.wav"

        if not vocals_path.exists():
            raise SeparationError(f"Vocals track not found: {vocals_path}")

        stems = {"vocals": vocals_path}

        if no_vocals_path.exists():
            stems["instrumental"] = no_vocals_path

        return stems

    def _organize_stems(self, stems: Dict[str, Path]) -> None:
        """
        Move stems to final location and update context.

        Args:
            stems: Dictionary with stem paths
        """
        if "vocals" in stems:
            vocals_src = stems["vocals"]
            vocals_dst = self.context.vocal_stem_path

            if vocals_dst:
                vocals_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(vocals_src), str(vocals_dst))
                self.logger.debug("Vocal track moved", destination=str(vocals_dst))

        if "instrumental" in stems:
            inst_src = stems["instrumental"]
            inst_dst = self.context.instrumental_stem_path

            if inst_dst:
                inst_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(inst_src), str(inst_dst))
                self.logger.debug("Instrumental track moved", destination=str(inst_dst))

        # Clean up Demucs temporary directory
        self._cleanup_demucs_output()

    def _cleanup_demucs_output(self) -> None:
        """Clean up Demucs temporary output directory."""
        try:
            output_dir = self.context.get_path("stems")
            model = self.config.get("model", "htdemucs")
            model_dir = output_dir / model

            if model_dir.exists():
                shutil.rmtree(model_dir)
                self.logger.debug("Demucs temporary directory cleaned", path=str(model_dir))
        except Exception as e:
            self.logger.warning(f"Failed to clean up Demucs output: {e}")


def create_separator(
    context: ProcessingContext,
    logger: ContextualLogger,
    gpu_manager: GPUManager,
) -> Separator:
    """
    Factory function to create a Separator instance.

    Args:
        context: Processing context
        logger: Contextual logger
        gpu_manager: GPU manager

    Returns:
        Separator instance
    """
    return Separator(context, logger, gpu_manager)
