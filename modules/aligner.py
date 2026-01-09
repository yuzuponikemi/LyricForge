"""
Forced Alignment Module

Aligns known lyrics text with audio using timing information from Whisper.
This is more accurate than pure transcription when lyrics are already known.
"""

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.context import ProcessingContext
from modules.transcriber import create_transcriber
from utils.gpu_manager import GPUManager
from utils.logger import ContextualLogger


class AlignmentError(Exception):
    """Exception raised when alignment fails."""

    pass


class Aligner:
    """
    Handles forced alignment of known lyrics with audio.
    """

    def __init__(
        self,
        context: ProcessingContext,
        logger: ContextualLogger,
        gpu_manager: GPUManager,
    ):
        """
        Initialize aligner.

        Args:
            context: Processing context
            logger: Contextual logger
            gpu_manager: GPU manager
        """
        self.context = context
        self.logger = logger
        self.gpu_manager = gpu_manager
        self.config = context.config.get("aligner", {})

    def align(
        self,
        lyrics_text: str,
        audio_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Align known lyrics with audio.

        Args:
            lyrics_text: Known lyrics text
            audio_path: Audio file path (uses context if not provided)

        Returns:
            Dictionary with aligned segments

        Raises:
            AlignmentError: If alignment fails
        """
        if audio_path is None:
            # Use vocal stem if available
            if self.context.vocal_stem_path and self.context.vocal_stem_path.exists():
                audio_path = self.context.vocal_stem_path
            else:
                audio_path = self.context.raw_audio_path

        if audio_path is None or not audio_path.exists():
            raise AlignmentError(f"Audio file not found: {audio_path}")

        self.logger.info("Starting forced alignment", audio_file=str(audio_path))

        try:
            # Step 1: Get timing from Whisper
            transcriber = create_transcriber(self.context, self.logger, self.gpu_manager)
            whisper_result = transcriber.transcribe(audio_path)

            # Load Whisper segments
            with open(self.context.raw_transcript_path, "r") as f:
                whisper_data = json.load(f)

            whisper_segments = whisper_data.get("segments", [])

            # Step 2: Align with known lyrics
            aligned_segments = self._align_lyrics_with_timing(
                lyrics_text, whisper_segments
            )

            # Step 3: Save aligned result
            self._save_aligned_transcription(aligned_segments)

            self.logger.info(
                "Forced alignment completed",
                segment_count=len(aligned_segments),
                output=str(self.context.refined_transcript_path),
            )

            return {
                "segments": aligned_segments,
                "path": self.context.refined_transcript_path,
            }

        except Exception as e:
            error_msg = f"Forced alignment failed: {str(e)}"
            self.logger.error(error_msg)
            raise AlignmentError(error_msg) from e

    def _align_lyrics_with_timing(
        self,
        lyrics_text: str,
        whisper_segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Align known lyrics with Whisper timing.

        Args:
            lyrics_text: Known lyrics
            whisper_segments: Whisper transcription segments

        Returns:
            Aligned segments with correct text and timing
        """
        # Split lyrics into lines
        lyrics_lines = [line.strip() for line in lyrics_text.split("\n") if line.strip()]

        # Extract Whisper text
        whisper_text = " ".join([seg.get("text", "").strip() for seg in whisper_segments])

        self.logger.debug(
            "Aligning lyrics",
            lyrics_lines=len(lyrics_lines),
            whisper_segments=len(whisper_segments),
        )

        # Use sequence matching to align lyrics with Whisper output
        aligned_segments = []

        # Simple strategy: distribute lyrics lines across Whisper segments
        if not whisper_segments:
            # No timing available, create segments with placeholder timing
            for i, line in enumerate(lyrics_lines):
                aligned_segments.append({
                    "id": i,
                    "start": i * 3.0,  # Placeholder: 3 seconds per line
                    "end": (i + 1) * 3.0,
                    "text": line,
                    "source": "manual",
                })
        else:
            # Match lyrics lines to Whisper segments using edit distance
            lyrics_to_segments = self._match_lyrics_to_segments(
                lyrics_lines, whisper_segments
            )

            for i, (line, segment) in enumerate(lyrics_to_segments):
                aligned_segments.append({
                    "id": i,
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": line,
                    "whisper_text": segment.get("text", ""),  # Keep original for reference
                    "confidence": self._calculate_match_confidence(line, segment.get("text", "")),
                    "source": "aligned",
                })

        return aligned_segments

    def _match_lyrics_to_segments(
        self,
        lyrics_lines: List[str],
        whisper_segments: List[Dict[str, Any]],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Match lyrics lines to Whisper segments.

        Args:
            lyrics_lines: List of lyrics lines
            whisper_segments: List of Whisper segments

        Returns:
            List of (lyrics_line, whisper_segment) pairs
        """
        matched_pairs = []

        # Calculate similarity matrix
        similarity_matrix = []
        for lyrics_line in lyrics_lines:
            row = []
            for segment in whisper_segments:
                whisper_text = segment.get("text", "").strip()
                similarity = self._text_similarity(lyrics_line, whisper_text)
                row.append(similarity)
            similarity_matrix.append(row)

        # Greedy matching: for each lyrics line, find best matching segment
        used_segments = set()

        for i, lyrics_line in enumerate(lyrics_lines):
            best_similarity = 0
            best_segment_idx = None

            for j, similarity in enumerate(similarity_matrix[i]):
                if j not in used_segments and similarity > best_similarity:
                    best_similarity = similarity
                    best_segment_idx = j

            if best_segment_idx is not None:
                matched_pairs.append((lyrics_line, whisper_segments[best_segment_idx]))
                used_segments.add(best_segment_idx)
            else:
                # No good match, use interpolated timing
                if matched_pairs:
                    last_end = matched_pairs[-1][1]["end"]
                    matched_pairs.append((
                        lyrics_line,
                        {
                            "start": last_end,
                            "end": last_end + 3.0,
                            "text": "",
                        },
                    ))
                else:
                    matched_pairs.append((
                        lyrics_line,
                        {
                            "start": i * 3.0,
                            "end": (i + 1) * 3.0,
                            "text": "",
                        },
                    ))

        return matched_pairs

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Normalize
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()

        if not text1 or not text2:
            return 0.0

        # Use SequenceMatcher
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio()

    def _calculate_match_confidence(self, lyrics_text: str, whisper_text: str) -> float:
        """
        Calculate confidence of the match.

        Args:
            lyrics_text: Known lyrics
            whisper_text: Whisper transcription

        Returns:
            Confidence score (0-1)
        """
        return self._text_similarity(lyrics_text, whisper_text)

    def _save_aligned_transcription(self, segments: List[Dict[str, Any]]) -> None:
        """
        Save aligned transcription to file.

        Args:
            segments: Aligned segments
        """
        output_path = self.context.refined_transcript_path

        if output_path is None:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "segments": segments,
            "metadata": {
                "method": "forced_alignment",
                "segment_count": len(segments),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.debug("Aligned transcription saved", path=str(output_path))


def create_aligner(
    context: ProcessingContext,
    logger: ContextualLogger,
    gpu_manager: GPUManager,
) -> Aligner:
    """
    Factory function to create an Aligner instance.

    Args:
        context: Processing context
        logger: Contextual logger
        gpu_manager: GPU manager

    Returns:
        Aligner instance
    """
    return Aligner(context, logger, gpu_manager)
