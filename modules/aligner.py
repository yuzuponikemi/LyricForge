"""
Forced Alignment Module

Aligns known lyrics text with audio using timing information from Whisper.
This is more accurate than pure transcription when lyrics are already known.
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.context import ProcessingContext
from modules.transcriber import create_transcriber
from utils.gpu_manager import GPUManager
from utils.logger import ContextualLogger

# Pattern to match section markers like [Verse 1], [Chorus], [Refrain], etc.
SECTION_MARKER_PATTERN = re.compile(
    r'^\s*\[(?:Verse|Chorus|Refrain|Bridge|Intro|Outro|Hook|Pre-Chorus|Post-Chorus|Interlude|Break|Solo|Instrumental|Tag|Coda|Ad[- ]?lib|Repeat|\d+|[A-Za-z]+\s*\d*)\s*\d*\]\s*$',
    re.IGNORECASE
)


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

    def _run_vocal_separation(self) -> Optional[Path]:
        """
        Run vocal separation to get cleaner audio for alignment.

        Returns:
            Path to vocal stem, or None if separation fails
        """
        try:
            from modules.separator import create_separator

            self.logger.info("Running vocal separation for better alignment")
            separator = create_separator(self.context, self.logger, self.gpu_manager)
            result = separator.separate()

            vocal_path = result.get("vocals")
            if vocal_path and vocal_path.exists():
                self.logger.info("Vocal separation completed", path=str(vocal_path))
                return vocal_path
            else:
                self.logger.warning("Vocal separation did not produce output, using raw audio")
                return self.context.raw_audio_path

        except Exception as e:
            self.logger.warning(f"Vocal separation failed, using raw audio: {e}")
            return self.context.raw_audio_path

    def _is_section_marker(self, line: str) -> bool:
        """
        Check if a line is a section marker like [Verse 1], [Chorus], etc.

        Args:
            line: Line to check

        Returns:
            True if line is a section marker
        """
        return bool(SECTION_MARKER_PATTERN.match(line.strip()))

    def _filter_lyrics_lines(self, lyrics_text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Filter lyrics text into actual lyrics and section markers.

        Args:
            lyrics_text: Raw lyrics text

        Returns:
            Tuple of (lyrics_lines, section_markers)
            section_markers contains {index, text} for reinsertion
        """
        all_lines = [line.strip() for line in lyrics_text.split("\n")]
        lyrics_lines = []
        section_markers = []

        for i, line in enumerate(all_lines):
            if not line:
                continue
            if self._is_section_marker(line):
                section_markers.append({"original_index": i, "text": line})
                self.logger.debug(f"Filtered section marker: {line}")
            else:
                lyrics_lines.append(line)

        return lyrics_lines, section_markers

    def align(
        self,
        lyrics_text: str,
        audio_path: Optional[Path] = None,
        run_separation: bool = True,
    ) -> Dict[str, Any]:
        """
        Align known lyrics with audio.

        Args:
            lyrics_text: Known lyrics text
            audio_path: Audio file path (uses context if not provided)
            run_separation: Whether to run vocal separation if no vocal stem exists

        Returns:
            Dictionary with aligned segments

        Raises:
            AlignmentError: If alignment fails
        """
        if audio_path is None:
            # Use vocal stem if available
            if self.context.vocal_stem_path and self.context.vocal_stem_path.exists():
                audio_path = self.context.vocal_stem_path
                self.logger.info("Using existing vocal stem", path=str(audio_path))
            elif run_separation and self.context.raw_audio_path:
                # Run vocal separation first for better alignment
                audio_path = self._run_vocal_separation()
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
        # Filter out section markers and get clean lyrics lines
        lyrics_lines, section_markers = self._filter_lyrics_lines(lyrics_text)

        if section_markers:
            self.logger.info(
                f"Filtered {len(section_markers)} section markers from lyrics",
                markers=[m["text"] for m in section_markers],
            )

        self.logger.debug(
            "Aligning lyrics",
            lyrics_lines=len(lyrics_lines),
            whisper_segments=len(whisper_segments),
        )

        aligned_segments = []

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
            # Use order-preserving matching algorithm
            lyrics_to_segments = self._match_lyrics_to_segments_ordered(
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

    def _match_lyrics_to_segments_ordered(
        self,
        lyrics_lines: List[str],
        whisper_segments: List[Dict[str, Any]],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Match lyrics lines to Whisper segments while preserving order.

        Uses dynamic programming to find the optimal alignment that
        maintains the temporal order of both lyrics and audio segments.

        Args:
            lyrics_lines: List of lyrics lines
            whisper_segments: List of Whisper segments (sorted by time)

        Returns:
            List of (lyrics_line, whisper_segment) pairs in order
        """
        n_lyrics = len(lyrics_lines)
        n_segments = len(whisper_segments)

        if n_lyrics == 0:
            return []

        if n_segments == 0:
            # No Whisper segments, create placeholder timing
            return [
                (line, {"start": i * 3.0, "end": (i + 1) * 3.0, "text": ""})
                for i, line in enumerate(lyrics_lines)
            ]

        # Sort segments by start time to ensure order
        sorted_segments = sorted(whisper_segments, key=lambda s: s.get("start", 0))

        # Calculate similarity matrix
        similarity_matrix = []
        for lyrics_line in lyrics_lines:
            row = []
            for segment in sorted_segments:
                whisper_text = segment.get("text", "").strip()
                similarity = self._text_similarity(lyrics_line, whisper_text)
                row.append(similarity)
            similarity_matrix.append(row)

        # Dynamic programming to find best order-preserving alignment
        # dp[i][j] = best score for aligning lyrics[0:i] with segments[0:j]
        # We allow skipping segments but not lyrics lines
        INF = float('-inf')
        dp = [[INF] * (n_segments + 1) for _ in range(n_lyrics + 1)]
        parent = [[None] * (n_segments + 1) for _ in range(n_lyrics + 1)]

        dp[0][0] = 0

        for i in range(n_lyrics + 1):
            for j in range(n_segments + 1):
                if dp[i][j] == INF:
                    continue

                # Option 1: Skip this segment (if j < n_segments)
                if j < n_segments and dp[i][j + 1] < dp[i][j]:
                    dp[i][j + 1] = dp[i][j]
                    parent[i][j + 1] = (i, j, "skip")

                # Option 2: Match lyrics[i] with segment[j] (if both available)
                if i < n_lyrics and j < n_segments:
                    score = dp[i][j] + similarity_matrix[i][j]
                    if score > dp[i + 1][j + 1]:
                        dp[i + 1][j + 1] = score
                        parent[i + 1][j + 1] = (i, j, "match")

        # Find best ending position (all lyrics must be matched)
        best_j = 0
        best_score = INF
        for j in range(n_segments + 1):
            if dp[n_lyrics][j] > best_score:
                best_score = dp[n_lyrics][j]
                best_j = j

        # Backtrack to find the alignment
        matched_pairs = []
        i, j = n_lyrics, best_j

        while i > 0 or j > 0:
            if parent[i][j] is None:
                break
            pi, pj, action = parent[i][j]
            if action == "match":
                matched_pairs.append((lyrics_lines[pi], sorted_segments[pj]))
            i, j = pi, pj

        matched_pairs.reverse()

        # If some lyrics weren't matched (shouldn't happen but handle gracefully)
        if len(matched_pairs) < n_lyrics:
            self.logger.warning(
                f"Only matched {len(matched_pairs)}/{n_lyrics} lyrics lines, "
                "using interpolation for remaining"
            )
            matched_set = set(line for line, _ in matched_pairs)
            last_end = matched_pairs[-1][1]["end"] if matched_pairs else 0

            for line in lyrics_lines:
                if line not in matched_set:
                    matched_pairs.append((
                        line,
                        {"start": last_end, "end": last_end + 3.0, "text": ""},
                    ))
                    last_end += 3.0

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
