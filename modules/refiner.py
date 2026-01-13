"""
LLM-based Lyric Refinement Module

Wrapper around Ollama API for refining raw transcriptions.
Corrects errors, improves formatting, and maintains timestamp alignment.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from core.context import ProcessingContext
from utils.logger import ContextualLogger


class RefinementError(Exception):
    """Exception raised when refinement fails."""

    pass


class Refiner:
    """
    Handles lyric refinement using Ollama LLM.
    """

    def __init__(
        self,
        context: ProcessingContext,
        logger: ContextualLogger,
    ):
        """
        Initialize refiner.

        Args:
            context: Processing context
            logger: Contextual logger
        """
        self.context = context
        self.logger = logger
        self.config = context.config.get("refiner", {})
        self.base_url = self.config.get("base_url", "http://localhost:11434")

    def refine(self, transcript_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Refine transcription using LLM.

        Args:
            transcript_path: Path to raw transcript (uses context if not provided)

        Returns:
            Dictionary with refined segments

        Raises:
            RefinementError: If refinement fails
        """
        if not self.config.get("enabled", True):
            self.logger.warning("Refiner is disabled in config, skipping")
            return {}

        if transcript_path is None:
            transcript_path = self.context.raw_transcript_path

        if transcript_path is None or not transcript_path.exists():
            raise RefinementError(f"Transcript file not found: {transcript_path}")

        self.logger.info("Starting lyric refinement", transcript=str(transcript_path))

        try:
            # Load raw transcription
            with open(transcript_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            segments = raw_data.get("segments", [])

            # Check Ollama availability
            if not self._check_ollama_available():
                raise RefinementError(
                    f"Ollama not available at {self.base_url}. "
                    "Make sure Ollama is running: ollama serve"
                )

            # Refine segments
            refined_segments = self._refine_segments(segments)

            # Save refined transcription
            self._save_refined_transcription(refined_segments)

            self.logger.info(
                "Lyric refinement completed",
                segment_count=len(refined_segments),
                output=str(self.context.refined_transcript_path),
            )

            return {
                "segments": refined_segments,
                "path": self.context.refined_transcript_path,
            }

        except Exception as e:
            error_msg = f"Lyric refinement failed: {str(e)}"
            self.logger.error(error_msg)
            raise RefinementError(error_msg) from e

    def _check_ollama_available(self) -> bool:
        """
        Check if Ollama server is available.

        Returns:
            True if Ollama is available
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _refine_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Refine transcription segments using LLM.

        Args:
            segments: List of raw segments

        Returns:
            List of refined segments
        """
        model = self.config.get("model", "llama3.2:latest")

        self.logger.info(
            "Refining segments with LLM",
            model=model,
            segment_count=len(segments),
        )

        # Group segments for context (process in batches to maintain context)
        batch_size = 10
        refined_segments = []

        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            refined_batch = self._refine_batch(batch, model)
            refined_segments.extend(refined_batch)

        return refined_segments

    def _refine_batch(
        self, segments: List[Dict[str, Any]], model: str
    ) -> List[Dict[str, Any]]:
        """
        Refine a batch of segments.

        Args:
            segments: List of segments to refine
            model: Ollama model name

        Returns:
            List of refined segments
        """
        # Build prompt
        prompt = self._build_prompt(segments)

        # Call Ollama API
        try:
            response = self._call_ollama(model, prompt)
            refined_segments = self._parse_llm_response(response, segments)
            return refined_segments

        except Exception as e:
            self.logger.warning(
                f"Failed to refine batch, using original text: {e}"
            )
            # Return original segments if refinement fails
            return segments

    def _build_prompt(self, segments: List[Dict[str, Any]]) -> str:
        """
        Build prompt for LLM refinement.

        Args:
            segments: List of segments

        Returns:
            Prompt string
        """
        template = self.config.get(
            "prompt_template",
            """You are an expert in transcribing song lyrics. Given the following raw transcription segments from Whisper, please:
1. Fix any obvious transcription errors (mishearings, wrong words)
2. Correct spelling and grammar while preserving the original meaning
3. For Japanese text, fix kanji/kana if needed
4. Remove filler words or non-lyrical sounds (like "[Music]", "(applause)", etc.)
5. Keep the text natural and lyrical

IMPORTANT: Return ONLY the corrected text for each segment, one per line, in the same order. Do not add explanations or change the number of segments.

Raw segments:
{transcription}

Corrected lyrics (one line per segment):""",
        )

        # Format segments for prompt
        segment_texts = []
        for seg in segments:
            segment_texts.append(f"{seg['id']}: {seg['text']}")

        transcription_text = "\n".join(segment_texts)

        return template.format(transcription=transcription_text)

    def _call_ollama(self, model: str, prompt: str) -> str:
        """
        Call Ollama API.

        Args:
            model: Model name
            prompt: Prompt text

        Returns:
            Response text

        Raises:
            RefinementError: If API call fails
        """
        url = f"{self.base_url}/api/generate"
        temperature = self.config.get("temperature", 0.3)
        max_tokens = self.config.get("max_tokens", 2000)
        timeout = self.config.get("timeout", 60)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

        except requests.exceptions.RequestException as e:
            raise RefinementError(f"Ollama API call failed: {str(e)}") from e

    def _parse_llm_response(
        self, response: str, original_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Parse LLM response and align with original timestamps.

        Args:
            response: LLM response text
            original_segments: Original segments with timestamps

        Returns:
            Refined segments with preserved timestamps
        """
        # Split response into lines
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]

        # Remove segment IDs if present (e.g., "0: text" -> "text")
        cleaned_lines = []
        for line in lines:
            if ":" in line and line.split(":")[0].strip().isdigit():
                # Remove "ID: " prefix
                cleaned_lines.append(":".join(line.split(":")[1:]).strip())
            else:
                cleaned_lines.append(line)

        # Align with original segments
        refined_segments = []
        for i, original in enumerate(original_segments):
            refined_text = cleaned_lines[i] if i < len(cleaned_lines) else original["text"]

            # Create refined segment with preserved timestamps
            refined_segment = {
                "id": original["id"],
                "start": original["start"],
                "end": original["end"],
                "text": refined_text,
                "original_text": original["text"],  # Keep original for reference
            }

            # Preserve word-level timestamps if available
            if "words" in original:
                refined_segment["words"] = original["words"]

            refined_segments.append(refined_segment)

        return refined_segments

    def _save_refined_transcription(self, segments: List[Dict[str, Any]]) -> None:
        """
        Save refined transcription to file.

        Args:
            segments: Refined segments
        """
        output_path = self.context.refined_transcript_path

        if output_path is None:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "segments": segments,
            "metadata": {
                "model": self.config.get("model", "llama3.2:latest"),
                "base_url": self.base_url,
                "segment_count": len(segments),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.debug("Refined transcription saved", path=str(output_path))


def create_refiner(
    context: ProcessingContext,
    logger: ContextualLogger,
) -> Refiner:
    """
    Factory function to create a Refiner instance.

    Args:
        context: Processing context
        logger: Contextual logger

    Returns:
        Refiner instance
    """
    return Refiner(context, logger)
