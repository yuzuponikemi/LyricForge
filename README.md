# 🔨 LyricForge - The Lyric Foundry

**Transform videos into beautifully subtitled content with AI-powered transcription.**

LyricForge is a modular, production-grade application for extracting, refining, and embedding lyrics/subtitles into videos. Built with a Pipeline Pattern architecture for maximum extensibility and maintainability.

## ✨ Features

### Current (Milestone 1: The Backbone) ✅

- **Video/Audio Download**: Support for YouTube, Vimeo, and 1000+ platforms via yt-dlp
- **AI Transcription**: State-of-the-art speech recognition using Faster-Whisper
- **Word-level Timestamps**: Precise timing for each word
- **GPU Acceleration**: Automatic device selection (CUDA/MPS/CPU) with memory management
- **Modular Architecture**: Clean separation of concerns for easy extension

### Upcoming

- **Milestone 2**: Vocal separation (Demucs) + LLM refinement (Ollama)
- **Milestone 3**: ASS subtitle styling + FFmpeg video composition
- **Milestone 4**: Obsidian knowledge base integration

## 🏗️ Architecture

LyricForge implements a **Pipeline Pattern** with independent, composable modules:

```
┌─────────────┐    ┌───────────┐    ┌──────────────┐    ┌──────────┐
│  Downloader │ -> │ Separator │ -> │ Transcriber  │ -> │ Refiner  │
│   (yt-dlp)  │    │ (Demucs)  │    │  (Whisper)   │    │ (Ollama) │
└─────────────┘    └───────────┘    └──────────────┘    └──────────┘
                                                               │
                                                               v
┌─────────────┐    ┌───────────────────────────────────────────┐
│   Output    │ <- │         Compositor (FFmpeg)               │
│             │    │                                           │
└─────────────┘    └───────────────────────────────────────────┘
```

### Directory Structure

```
LyricForge/
├── config/              # Configuration files
│   └── settings.yaml    # Main settings (paths, models, parameters)
├── core/                # Core business logic
│   ├── context.py       # Execution context management
│   └── pipeline.py      # Pipeline orchestration
├── modules/             # Processing modules
│   ├── downloader.py    # yt-dlp wrapper
│   ├── separator.py     # Demucs wrapper (M2)
│   ├── transcriber.py   # Faster-Whisper wrapper
│   ├── refiner.py       # Ollama LLM client (M2)
│   └── compositor.py    # FFmpeg wrapper (M3)
├── utils/               # Utilities
│   ├── logger.py        # Structured logging
│   └── gpu_manager.py   # VRAM management
├── data/                # Generated content (gitignored)
│   ├── raw/             # Downloaded videos/audio
│   ├── stems/           # Separated audio stems
│   ├── subs/            # Generated subtitles
│   └── output/          # Final videos
└── main.py              # CLI entry point
```

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- FFmpeg (for audio/video processing)
- CUDA-compatible GPU (optional, for acceleration)

### Step 1: Clone and Install Dependencies

```bash
git clone https://github.com/yourusername/LyricForge.git
cd LyricForge

# Option A: Using pip
pip install -r requirements.txt

# Option B: Using poetry/uv (recommended for development)
uv pip install -e .
```

### Step 2: Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Step 3: Verify Installation

```bash
python main.py info
```

This will display your system information and available compute devices.

## 🚀 Quick Start

### Basic Usage

```bash
# Process a YouTube video (download + transcribe)
python main.py process "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Advanced Usage

```bash
# Specify language and model size
python main.py process "URL" --language ja --model large-v3

# Use CPU only (if you don't have a GPU)
python main.py process "URL" --device cpu

# Run specific stages
python main.py process "URL" --stages download --stages transcribe

# Use custom config file
python main.py process "URL" --config my_config.yaml
```

### Output

After processing, you'll find:
- `data/raw/`: Original video and audio files
- `data/subs/`: Transcription JSON with timestamps
- Logs: Console output with processing details

## ⚙️ Configuration

Edit `config/settings.yaml` to customize:

### Key Settings

```yaml
# Transcription settings
transcriber:
  model: "large-v3"        # tiny, base, small, medium, large-v2, large-v3
  language: "ja"           # ISO 639-1 code, null for auto-detection
  device: "auto"           # auto, cuda, mps, cpu
  vad_filter: true         # Voice Activity Detection

# GPU settings
gpu:
  max_vram_usage: 0.8      # Use up to 80% of available VRAM
  force_cpu: false
  prefer_mps: true         # Use Metal on macOS

# Logging
logging:
  level: "INFO"            # DEBUG, INFO, WARNING, ERROR
  rich_console: true       # Colored output
```

See `config/settings.yaml` for all available options.

## 🧪 Development

### Project Structure

- **core/**: Business logic that doesn't depend on specific technologies
- **modules/**: Technology-specific implementations (yt-dlp, Whisper, etc.)
- **utils/**: Shared utilities (logging, GPU management)

### Adding a New Module

1. Create a new file in `modules/`
2. Implement the processor class
3. Add a factory function
4. Register in `core/pipeline.py`

Example:

```python
# modules/my_processor.py
class MyProcessor:
    def __init__(self, context, logger):
        self.context = context
        self.logger = logger

    def process(self):
        # Your processing logic
        pass

def create_my_processor(context, logger):
    return MyProcessor(context, logger)
```

## 🗺️ Roadmap

### ✅ Milestone 1: The Backbone (Current)
- [x] Video download (yt-dlp)
- [x] Audio transcription (Faster-Whisper)
- [x] Modular architecture
- [x] GPU management
- [x] CLI interface

### 🚧 Milestone 2: The Ear & The Brain
- [ ] Vocal separation (Demucs)
- [ ] LLM refinement (Ollama)
- [ ] Timestamp alignment
- [ ] Multi-language support enhancement

### 📋 Milestone 3: The Artist
- [ ] ASS subtitle generation with styling
- [ ] FFmpeg video composition
- [ ] GPU-accelerated encoding
- [ ] Custom subtitle templates

### 🗂️ Milestone 4: The Archivist
- [ ] Obsidian integration
- [ ] Markdown note generation
- [ ] Metadata extraction
- [ ] Knowledge base indexing

## 🔧 Troubleshooting

### "CUDA out of memory" Error

1. Reduce model size: Use `--model medium` or `--model small`
2. Force CPU: Use `--device cpu`
3. Adjust VRAM limit in `config/settings.yaml`:
   ```yaml
   gpu:
     max_vram_usage: 0.6  # Reduce to 60%
   ```

### "Model not found" Error

Faster-Whisper downloads models on first use. Ensure you have internet connection and sufficient disk space (~3GB for large-v3).

### "FFmpeg not found" Error

Make sure FFmpeg is installed and available in your PATH:
```bash
ffmpeg -version
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Universal video downloader
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) - Fast Whisper implementation
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [Demucs](https://github.com/facebookresearch/demucs) - Audio source separation
- [Ollama](https://ollama.ai/) - Local LLM inference

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ for creators who believe in local-first, privacy-preserving AI tools.**
