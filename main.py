#!/usr/bin/env python3
"""
LyricForge - Main CLI Entry Point

Command-line interface for the LyricForge pipeline.
"""

import sys
from pathlib import Path
from typing import List, Optional

import click

from core.pipeline import run_pipeline


@click.group()
@click.version_option(version="0.3.0", prog_name="LyricForge")
def cli():
    """
    LyricForge - The Lyric Foundry

    Transform videos into beautifully subtitled content with AI-powered transcription.
    """
    pass


@cli.command()
@click.argument("url", type=str)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default="config/settings.yaml",
    help="Path to configuration file",
)
@click.option(
    "--stages",
    "-s",
    multiple=True,
    type=click.Choice(["download", "calibrate", "separate", "transcribe", "align", "refine", "subtitle", "compose"]),
    help="Stages to execute (can be specified multiple times)",
)
@click.option(
    "--language",
    "-l",
    type=str,
    help="Language code (e.g., 'en', 'ja', 'zh') - overrides config",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["tiny", "base", "small", "medium", "large-v2", "large-v3"]),
    help="Whisper model size - overrides config",
)
@click.option(
    "--device",
    "-d",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    help="Compute device - overrides config",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory - overrides config",
)
@click.option(
    "--auto-calibrate",
    "-a",
    is_flag=True,
    help="Automatically calibrate parameters by analyzing a 30s sample",
)
@click.option(
    "--lyrics-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to lyrics text file for forced alignment",
)
@click.option(
    "--artist",
    type=str,
    help="Artist name (for API lyrics search)",
)
@click.option(
    "--song-title",
    type=str,
    help="Song title (for API lyrics search)",
)
def process(
    url: str,
    config: Path,
    stages: tuple,
    language: Optional[str],
    model: Optional[str],
    device: Optional[str],
    output_dir: Optional[Path],
    auto_calibrate: bool,
    lyrics_file: Optional[Path],
    artist: Optional[str],
    song_title: Optional[str],
):
    """
    Process a video URL through the LyricForge pipeline.

    URL: Video URL from YouTube, Vimeo, or other supported platforms.

    Examples:

        # Basic usage (download + transcribe)
        lyric-forge process "https://www.youtube.com/watch?v=VIDEO_ID"

        # Auto-calibrate for optimal quality (recommended for live/noisy audio)
        lyric-forge process "URL" --auto-calibrate

        # Specify language and model
        lyric-forge process "URL" --language ja --model large-v3

        # Run specific stages
        lyric-forge process "URL" --stages download --stages transcribe

        # Use CPU only
        lyric-forge process "URL" --device cpu
    """
    try:
        # Convert stages tuple to list
        stages_list = list(stages) if stages else None

        # Set default stages for forced alignment mode
        if lyrics_file or artist or song_title:
            if stages_list is None:
                # Forced alignment mode: download + align (align includes separation internally)
                stages_list = ["download", "align"]
                click.echo("🎵 Forced alignment mode enabled")

        # Add calibration stage if auto-calibrate is enabled
        if auto_calibrate:
            if stages_list is None:
                # Insert calibrate after download in default pipeline
                stages_list = ["download", "calibrate", "separate", "transcribe", "refine"]
            elif "calibrate" not in stages_list and "download" in stages_list:
                # Insert calibrate after download
                download_idx = stages_list.index("download")
                stages_list.insert(download_idx + 1, "calibrate")

        # Build config overrides
        config_overrides = {}
        if language:
            config_overrides.setdefault("transcriber", {})["language"] = language
        if model:
            config_overrides.setdefault("transcriber", {})["model"] = model
        if device:
            # Override device for both transcriber and separator if needed
            config_overrides.setdefault("transcriber", {})["device"] = device
            config_overrides.setdefault("separator", {})["device"] = device
            # Also override global gpu settings
            if device == "cpu":
                config_overrides.setdefault("gpu", {})["force_cpu"] = True
                config_overrides.setdefault("gpu", {})["prefer_mps"] = False
                config_overrides.setdefault("transcriber", {})["compute_type"] = "int8"

        if output_dir:
            config_overrides.setdefault("paths", {})["output"] = str(output_dir)

        # Display info
        click.echo(f"🔨 LyricForge - Processing: {url}")
        if stages_list:
            click.echo(f"📋 Stages: {', '.join(stages_list)}")
        click.echo()

        # Run pipeline
        context = run_pipeline(
            url=url,
            config_path=str(config),
            stages=stages_list,
            config_overrides=config_overrides,
            lyrics_file=lyrics_file,
            artist=artist,
            song_title=song_title,
        )

        # Display results
        click.echo()
        click.echo("✅ Processing completed successfully!")
        click.echo(f"⏱️  Duration: {context.get_duration():.2f}s")
        click.echo()
        click.echo("📁 Output files:")

        if context.raw_video_path and context.raw_video_path.exists():
            click.echo(f"   Video: {context.raw_video_path}")
        if context.raw_audio_path and context.raw_audio_path.exists():
            click.echo(f"   Audio: {context.raw_audio_path}")
        if context.vocal_stem_path and context.vocal_stem_path.exists():
            click.echo(f"   Vocals: {context.vocal_stem_path}")
        if context.raw_transcript_path and context.raw_transcript_path.exists():
            click.echo(f"   Transcript: {context.raw_transcript_path}")
        if context.refined_transcript_path and context.refined_transcript_path.exists():
            click.echo(f"   Aligned: {context.refined_transcript_path}")
        if context.subtitle_path and context.subtitle_path.exists():
            click.echo(f"   Subtitles: {context.subtitle_path}")
        if context.output_video_path and context.output_video_path.exists():
            click.echo(f"   Output: {context.output_video_path}")

    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default="config/settings.yaml",
    help="Path to configuration file",
)
def info(config: Path):
    """
    Display system and GPU information.
    """
    try:
        from core.context import ConfigLoader
        from utils.gpu_manager import create_gpu_manager, print_device_info

        # Load config
        config_dict = ConfigLoader.load_config(str(config))

        # Create GPU manager
        gpu_manager = create_gpu_manager(config_dict)

        # Print info
        print_device_info(gpu_manager)

    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def version():
    """
    Display version information.
    """
    click.echo("LyricForge v0.3.0")
    click.echo("The Lyric Foundry - AI-Powered Video Transcription")
    click.echo()
    click.echo("Milestone 1: The Backbone ✅")
    click.echo("  - Download (yt-dlp)")
    click.echo("  - Transcription (Faster-Whisper)")
    click.echo()
    click.echo("Milestone 2: The Ear & The Brain ✅")
    click.echo("  - Vocal separation (Demucs)")
    click.echo("  - LLM refinement (Ollama)")
    click.echo()
    click.echo("Milestone 2.5: Forced Alignment & Auto-Calibration ✅")
    click.echo("  - Auto-calibration for optimal parameters")
    click.echo("  - Lyrics fetching from file or API")
    click.echo("  - Forced alignment with known lyrics")
    click.echo()
    click.echo("Upcoming:")
    click.echo("  - Milestone 3: ASS subtitles + FFmpeg composition")
    click.echo("  - Milestone 4: Obsidian integration")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
