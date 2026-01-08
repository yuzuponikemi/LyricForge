"""
Audio Source Separation Module

Wrapper around Demucs for separating audio into stems (vocals, drums, bass, other).
Optimized for lyric extraction by isolating vocal tracks.
Uses Demucs Python API with soundfile for saving to avoid torchaudio/torchcodec issues.
"""

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch

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
        Separate audio using Demucs Python API.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with stem paths
        """
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        model_name = self.config.get("model", "htdemucs")
        device = self._get_device()
        shifts = self.config.get("shifts", 1)
        split = self.config.get("split", True)
        overlap = self.config.get("overlap", 0.25)

        # Output directory
        output_dir = self.context.get_path("stems")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Running Demucs",
            model=model_name,
            device=device,
            shifts=shifts,
            split=split,
        )

        try:
            # Load model
            model = get_model(model_name)
            model.to(device)
            model.eval()

            # Load audio using soundfile to avoid torchaudio/torchcodec issues
            audio_data, sr = sf.read(str(audio_path))
            # Convert to torch tensor: soundfile returns (samples, channels), we need (channels, samples)
            if audio_data.ndim == 1:
                audio_data = audio_data[:, np.newaxis]
            wav = torch.from_numpy(audio_data.T).float()

            # Resample if needed using scipy to avoid torchaudio
            if sr != model.samplerate:
                from scipy import signal
                self.logger.debug(f"Resampling from {sr} to {model.samplerate}")
                # wav shape: (channels, samples)
                num_samples = int(wav.shape[1] * model.samplerate / sr)
                resampled = np.zeros((wav.shape[0], num_samples), dtype=np.float32)
                for ch in range(wav.shape[0]):
                    resampled[ch] = signal.resample(wav[ch].numpy(), num_samples)
                wav = torch.from_numpy(resampled)
                sr = model.samplerate

            # Ensure stereo
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2:
                wav = wav[:2]

            # Add batch dimension: (channels, samples) -> (batch, channels, samples)
            wav = wav.unsqueeze(0).to(device)

            # Apply model
            with torch.no_grad():
                sources = apply_model(
                    model,
                    wav,
                    shifts=shifts,
                    split=split,
                    overlap=overlap,
                    progress=True,
                )

            # Get source indices
            source_names = model.sources
            vocals_idx = source_names.index("vocals") if "vocals" in source_names else None

            if vocals_idx is None:
                raise SeparationError("Model does not have 'vocals' source")

            # Extract vocals and instrumental
            # sources shape: (batch, sources, channels, samples)
            vocals = sources[0, vocals_idx].cpu().numpy()

            # Create instrumental by summing all non-vocal sources
            other_indices = [i for i in range(len(source_names)) if i != vocals_idx]
            instrumental = sources[0, other_indices].sum(dim=0).cpu().numpy()

            # Save using soundfile (avoids torchaudio/torchcodec issues)
            audio_stem = audio_path.stem
            model_output = output_dir / model_name / audio_stem
            model_output.mkdir(parents=True, exist_ok=True)

            vocals_path = model_output / "vocals.wav"
            no_vocals_path = model_output / "no_vocals.wav"

            # Transpose from (channels, samples) to (samples, channels) for soundfile
            sf.write(str(vocals_path), vocals.T, sr)
            sf.write(str(no_vocals_path), instrumental.T, sr)

            self.logger.debug("Stems saved", vocals=str(vocals_path), instrumental=str(no_vocals_path))

        except Exception as e:
            raise SeparationError(f"Demucs separation failed: {e}") from e

        stems = {"vocals": vocals_path, "instrumental": no_vocals_path}

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
