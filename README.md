# whisper

Native R torch implementation of OpenAI Whisper for speech-to-text transcription.

## Installation

```r
# Install dependencies
install.packages(c("torch", "hfhub", "safetensors", "av", "jsonlite"))

# Install whisper from GitHub
remotes::install_github("cornball-ai/whisper")
```

## Quick Start

```r
library(whisper)

# Transcribe the bundled JFK "Ask not" speech (prompts to download model on first use)
jfk <- system.file("audio", "jfk.mp3", package = "whisper")
result <- transcribe(jfk)
result$text
#> "Ask not what your country can do for you, ask what you can do for your country."
```

On first use, you'll be prompted to download the model:

```
Download 'tiny' model (~151 MB) from HuggingFace? (Yes/no/cancel)
```

## Model Management

```r
# Download a model explicitly
download_whisper_model("tiny")

# List available models
list_whisper_models()
#> [1] "tiny" "base" "small" "medium" "large-v3"

# Check which models are downloaded
list_downloaded_models()

# Check if a specific model exists locally
model_exists("tiny")
```

## Usage

```r
# Basic transcription
result <- transcribe("audio.wav")
print(result$text)

# Specify model size
result <- transcribe("audio.wav", model = "small")

# Get word timestamps
result <- transcribe("audio.wav", timestamps = TRUE)
print(result$segments)

# Translate to English
result <- transcribe("audio.wav", task = "translate")
```

## Models

| Model | Parameters | Size | English WER |
|-------|------------|------|-------------|
| tiny | 39M | 151 MB | ~9% |
| base | 74M | 290 MB | ~7% |
| small | 244M | 967 MB | ~5% |
| medium | 769M | 3.0 GB | ~4% |
| large-v3 | 1550M | 6.2 GB | ~3% |

Models are downloaded from HuggingFace and cached in `~/.cache/huggingface/`.

## License

MIT
