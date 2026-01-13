"""
Audio Quality Analysis Utilities

Provides metrics and tools for analyzing audio quality to guide parameter calibration.
Includes SNR estimation, clarity metrics, and transcription quality scoring.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class AudioAnalyzer:
    """
    Analyzes audio quality for calibration purposes.
    """

    @staticmethod
    def calculate_snr(vocal_path: Path, instrumental_path: Optional[Path] = None) -> float:
        """
        Calculate Signal-to-Noise Ratio for separated audio.

        Args:
            vocal_path: Path to vocal track
            instrumental_path: Path to instrumental track (optional)

        Returns:
            SNR in dB (higher is better)
        """
        try:
            # Load audio using ffmpeg
            vocal_rms = AudioAnalyzer._calculate_rms(vocal_path)

            if instrumental_path and instrumental_path.exists():
                noise_rms = AudioAnalyzer._calculate_rms(instrumental_path)
            else:
                # Estimate noise from quiet sections of vocal track
                noise_rms = vocal_rms * 0.1  # Rough estimate

            if noise_rms == 0:
                return 60.0  # Very high SNR

            snr = 20 * np.log10(vocal_rms / noise_rms)
            return float(snr)

        except Exception:
            return 0.0  # Failed to calculate

    @staticmethod
    def _calculate_rms(audio_path: Path) -> float:
        """
        Calculate RMS (Root Mean Square) amplitude of audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            RMS value
        """
        try:
            # Use ffmpeg to get audio stats
            cmd = [
                "ffmpeg",
                "-i",
                str(audio_path),
                "-af",
                "volumedetect",
                "-f",
                "null",
                "-",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, stderr=subprocess.STDOUT
            )

            # Parse mean volume from output
            for line in result.stdout.split("\n"):
                if "mean_volume:" in line:
                    # Extract dB value
                    db_str = line.split("mean_volume:")[1].split("dB")[0].strip()
                    db_value = float(db_str)
                    # Convert dB to linear scale
                    return 10 ** (db_value / 20)

            return 1.0  # Default if parsing fails

        except Exception:
            return 1.0

    @staticmethod
    def score_transcription_quality(segments: List[Dict[str, Any]]) -> float:
        """
        Score transcription quality based on confidence and characteristics.

        Args:
            segments: Transcription segments with word-level data

        Returns:
            Quality score (0-100, higher is better)
        """
        if not segments:
            return 0.0

        scores = []

        # 1. Average word confidence
        total_confidence = 0.0
        word_count = 0

        for seg in segments:
            if "words" in seg:
                for word in seg["words"]:
                    if "probability" in word:
                        total_confidence += word["probability"]
                        word_count += 1

        avg_confidence = total_confidence / word_count if word_count > 0 else 0.5

        # 2. Segment duration consistency (prefer reasonable lengths)
        durations = []
        for seg in segments:
            duration = seg.get("end", 0) - seg.get("start", 0)
            durations.append(duration)

        if durations:
            avg_duration = np.mean(durations)
            duration_std = np.std(durations)

            # Ideal segment duration: 2-8 seconds
            duration_score = 1.0
            if avg_duration < 1.0:
                duration_score = 0.5  # Too short (fragmented)
            elif avg_duration > 15.0:
                duration_score = 0.6  # Too long (missed boundaries)

            # Penalize high variance
            if duration_std > 5.0:
                duration_score *= 0.8
        else:
            duration_score = 0.5

        # 3. Text quality (no repetitions, reasonable length)
        text_score = AudioAnalyzer._score_text_quality(segments)

        # Combine scores
        final_score = (avg_confidence * 50 + duration_score * 30 + text_score * 20)

        return float(final_score)

    @staticmethod
    def _score_text_quality(segments: List[Dict[str, Any]]) -> float:
        """
        Score text quality (detect repetitions, gibberish, etc.).

        Args:
            segments: Transcription segments

        Returns:
            Text quality score (0-1)
        """
        if not segments:
            return 0.0

        all_text = " ".join([seg.get("text", "") for seg in segments])

        # Check for excessive repetition
        words = all_text.split()
        if len(words) > 5:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)

            if repetition_ratio < 0.3:  # Too repetitive
                return 0.3
            elif repetition_ratio < 0.5:
                return 0.6
            else:
                return 1.0
        else:
            return 0.5  # Too short to judge

    @staticmethod
    def extract_sample(
        audio_path: Path, output_path: Path, duration: int = 30, offset: Optional[int] = None
    ) -> Path:
        """
        Extract a sample from audio file for calibration.

        Args:
            audio_path: Source audio file
            output_path: Output sample file
            duration: Sample duration in seconds
            offset: Start offset (None = center of file)

        Returns:
            Path to extracted sample

        Raises:
            RuntimeError: If extraction fails
        """
        try:
            # Get audio duration if offset is None
            if offset is None:
                total_duration = AudioAnalyzer._get_audio_duration(audio_path)
                offset = max(0, (total_duration - duration) // 2)

            # Extract sample using ffmpeg
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-i",
                str(audio_path),
                "-ss",
                str(offset),
                "-t",
                str(duration),
                "-c",
                "copy",
                str(output_path),
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )

            if not output_path.exists():
                raise RuntimeError("Failed to create sample file")

            return output_path

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg sample extraction failed: {e.stderr}") from e

    @staticmethod
    def _get_audio_duration(audio_path: Path) -> int:
        """
        Get audio file duration in seconds.

        Args:
            audio_path: Audio file path

        Returns:
            Duration in seconds
        """
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )

            duration = float(result.stdout.strip())
            return int(duration)

        except Exception:
            return 180  # Default: 3 minutes


def calculate_overall_score(
    snr: float,
    transcription_quality: float,
    segment_count: int,
    ideal_segment_count: int = 15,
) -> float:
    """
    Calculate overall quality score combining multiple metrics.

    Args:
        snr: Signal-to-Noise Ratio in dB
        transcription_quality: Transcription quality score (0-100)
        segment_count: Number of segments detected
        ideal_segment_count: Expected number of segments for sample duration

    Returns:
        Overall score (0-100)
    """
    # Normalize SNR (typical range: 0-30 dB)
    snr_normalized = min(snr / 30.0, 1.0) * 100

    # Segment count score (prefer close to ideal)
    if segment_count == 0:
        segment_score = 0.0
    else:
        ratio = segment_count / ideal_segment_count
        if 0.5 <= ratio <= 1.5:
            segment_score = 100.0
        elif 0.3 <= ratio <= 2.0:
            segment_score = 70.0
        else:
            segment_score = 40.0

    # Weighted combination
    overall = (
        snr_normalized * 0.3 + transcription_quality * 0.5 + segment_score * 0.2
    )

    return float(overall)
