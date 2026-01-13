"""
Auto-Calibration Module

Automatically tunes parameters by testing on a sample of the video.
Optimizes for clearest lyric extraction in challenging audio conditions
(live music, noisy environments, poor audio quality).
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.context import ProcessingContext
from modules.refiner import create_refiner
from modules.separator import create_separator
from modules.transcriber import create_transcriber
from utils.audio_analysis import AudioAnalyzer, calculate_overall_score
from utils.gpu_manager import GPUManager
from utils.logger import ContextualLogger


class CalibrationError(Exception):
    """Exception raised when calibration fails."""

    pass


class Calibrator:
    """
    Handles automatic parameter calibration.
    """

    # Predefined parameter presets for different scenarios
    PRESETS = {
        "studio": {
            "name": "Studio Recording",
            "separator": {
                "model": "htdemucs",
                "shifts": 1,
                "overlap": 0.25,
            },
            "transcriber": {
                "beam_size": 5,
                "vad_parameters": {
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                },
            },
        },
        "live": {
            "name": "Live Performance",
            "separator": {
                "model": "htdemucs",
                "shifts": 3,  # More shifts for better quality
                "overlap": 0.5,  # More overlap for stability
            },
            "transcriber": {
                "beam_size": 7,  # Larger beam for noisy audio
                "vad_parameters": {
                    "threshold": 0.6,  # Higher threshold for crowd noise
                    "min_speech_duration_ms": 300,
                },
            },
        },
        "noisy": {
            "name": "Noisy Environment",
            "separator": {
                "model": "htdemucs_ft",  # Fine-tuned model
                "shifts": 5,  # Maximum quality
                "overlap": 0.5,
            },
            "transcriber": {
                "beam_size": 10,  # Very large beam
                "vad_parameters": {
                    "threshold": 0.7,  # Strict VAD
                    "min_speech_duration_ms": 400,
                },
            },
        },
        "fast": {
            "name": "Fast Processing",
            "separator": {
                "model": "htdemucs",
                "shifts": 0,  # No shifts for speed
                "overlap": 0.25,
            },
            "transcriber": {
                "beam_size": 3,  # Smaller beam
                "vad_parameters": {
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                },
            },
        },
    }

    def __init__(
        self,
        context: ProcessingContext,
        logger: ContextualLogger,
        gpu_manager: GPUManager,
    ):
        """
        Initialize calibrator.

        Args:
            context: Processing context
            logger: Contextual logger
            gpu_manager: GPU manager
        """
        self.context = context
        self.logger = logger
        self.gpu_manager = gpu_manager
        self.config = context.config.get("calibrator", {})
        self.analyzer = AudioAnalyzer()

    def calibrate(
        self, audio_path: Optional[Path] = None, sample_duration: int = 30
    ) -> Dict[str, Any]:
        """
        Perform automatic calibration on a sample of the audio.

        Args:
            audio_path: Audio file to calibrate (uses context if not provided)
            sample_duration: Duration of sample in seconds

        Returns:
            Dictionary with optimal parameters and results

        Raises:
            CalibrationError: If calibration fails
        """
        if audio_path is None:
            audio_path = self.context.raw_audio_path

        if audio_path is None or not audio_path.exists():
            raise CalibrationError(f"Audio file not found: {audio_path}")

        self.logger.info(
            "Starting auto-calibration",
            audio_file=str(audio_path),
            sample_duration=sample_duration,
        )

        try:
            # Create temporary directory for calibration
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Extract sample from middle of audio
                sample_path = temp_path / "sample.wav"
                self.logger.info("Extracting audio sample for calibration")
                self.analyzer.extract_sample(
                    audio_path, sample_path, duration=sample_duration
                )

                # Test each preset
                results = self._test_presets(sample_path, temp_path)

                # Find best preset
                best_preset = self._select_best_preset(results)

                self.logger.info(
                    "Calibration completed",
                    best_preset=best_preset["name"],
                    score=f"{best_preset['score']:.2f}",
                )

                return best_preset

        except Exception as e:
            error_msg = f"Calibration failed: {str(e)}"
            self.logger.error(error_msg)
            raise CalibrationError(error_msg) from e

    def _test_presets(
        self, sample_path: Path, temp_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Test all parameter presets on the sample.

        Args:
            sample_path: Path to audio sample
            temp_dir: Temporary directory for processing

        Returns:
            List of results for each preset
        """
        results = []

        for preset_name, preset_params in self.PRESETS.items():
            self.logger.info(f"Testing preset: {preset_params['name']}")

            try:
                result = self._test_single_preset(
                    preset_name, preset_params, sample_path, temp_dir
                )
                results.append(result)

                self.logger.info(
                    f"Preset score",
                    preset=preset_params['name'],
                    score=f"{result['score']:.2f}",
                    snr=f"{result.get('snr', 0):.1f}dB",
                    quality=f"{result.get('transcription_quality', 0):.1f}",
                )

            except Exception as e:
                self.logger.warning(
                    f"Failed to test preset {preset_params['name']}: {e}"
                )
                # Add failed result with low score
                results.append({
                    "preset_name": preset_name,
                    "name": preset_params['name'],
                    "score": 0.0,
                    "error": str(e),
                    "parameters": preset_params,
                })

            # Clean up preset-specific files
            self._cleanup_preset_files(temp_dir, preset_name)

        return results

    def _test_single_preset(
        self,
        preset_name: str,
        preset_params: Dict[str, Any],
        sample_path: Path,
        temp_dir: Path,
    ) -> Dict[str, Any]:
        """
        Test a single parameter preset.

        Args:
            preset_name: Preset identifier
            preset_params: Parameter dictionary
            sample_path: Sample audio path
            temp_dir: Temporary directory

        Returns:
            Result dictionary with scores
        """
        # Create temporary context for this preset
        preset_dir = temp_dir / preset_name
        preset_dir.mkdir(exist_ok=True)

        # Build config with preset parameters
        test_config = self.context.config.copy()
        test_config["separator"].update(preset_params.get("separator", {}))
        test_config["transcriber"].update(preset_params.get("transcriber", {}))

        # Create temporary context
        test_context = ProcessingContext(
            url=self.context.url,
            video_id=f"calibration_{preset_name}",
            config=test_config,
        )

        # Set paths for this test
        test_context.raw_audio_path = sample_path
        test_context.vocal_stem_path = preset_dir / "vocals.wav"
        test_context.raw_transcript_path = preset_dir / "transcript.json"

        # Run separation
        separator = create_separator(test_context, self.logger, self.gpu_manager)
        sep_result = separator.separate(sample_path)

        # Calculate SNR
        snr = self.analyzer.calculate_snr(
            test_context.vocal_stem_path,
            sep_result.get("instrumental"),
        )

        # Run transcription
        transcriber = create_transcriber(
            test_context, self.logger, self.gpu_manager
        )
        trans_result = transcriber.transcribe(test_context.vocal_stem_path)

        # Load transcription and score
        with open(test_context.raw_transcript_path, "r") as f:
            trans_data = json.load(f)

        segments = trans_data.get("segments", [])
        transcription_quality = self.analyzer.score_transcription_quality(segments)

        # Calculate overall score
        overall_score = calculate_overall_score(
            snr=snr,
            transcription_quality=transcription_quality,
            segment_count=len(segments),
            ideal_segment_count=15,  # ~2s per segment for 30s sample
        )

        return {
            "preset_name": preset_name,
            "name": preset_params["name"],
            "score": overall_score,
            "snr": snr,
            "transcription_quality": transcription_quality,
            "segment_count": len(segments),
            "parameters": preset_params,
        }

    def _select_best_preset(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Select the best preset based on results.

        Args:
            results: List of test results

        Returns:
            Best preset result
        """
        if not results:
            # Fallback to studio preset
            return {
                "preset_name": "studio",
                "name": "Studio Recording (fallback)",
                "score": 0.0,
                "parameters": self.PRESETS["studio"],
            }

        # Sort by score
        sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)

        return sorted_results[0]

    def _cleanup_preset_files(self, temp_dir: Path, preset_name: str) -> None:
        """
        Clean up files from preset test.

        Args:
            temp_dir: Temporary directory
            preset_name: Preset identifier
        """
        preset_dir = temp_dir / preset_name
        if preset_dir.exists():
            try:
                shutil.rmtree(preset_dir)
            except Exception as e:
                self.logger.debug(f"Failed to cleanup preset dir: {e}")


def create_calibrator(
    context: ProcessingContext,
    logger: ContextualLogger,
    gpu_manager: GPUManager,
) -> Calibrator:
    """
    Factory function to create a Calibrator instance.

    Args:
        context: Processing context
        logger: Contextual logger
        gpu_manager: GPU manager

    Returns:
        Calibrator instance
    """
    return Calibrator(context, logger, gpu_manager)
