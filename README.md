# whisper

Native R torch implementation of OpenAI Whisper for speech-to-text transcription.

## Installation

```r
# Install dependencies
install.packages(c("torch", "av", "jsonlite"))

# Install whisper
remotes::install_local("~/whisper")
```

## Usage

```r
library(whisper)

# Basic transcription
result <- transcribe("audio.wav")
print(result$text)

# Specify model size
result <- transcribe("audio.wav", model = "small")

# Get word timestamps
result <- transcribe("audio.wav", timestamps = TRUE)
print(result$segments)
```

## Models

| Model | Parameters | Size (GB) | English Word Error Rate |
|-------|------------|-----------|------------------------|
| tiny | 39M | 0.14 | ~9% |
| base | 74M | 0.27 | ~7% |
| small | 244M | 0.9 | ~5% |
| medium | 769M | 2.9 | ~4% |
| large-v3 | 1550M | 2.9 | ~3% |

Models are automatically downloaded from HuggingFace on first use.

## License

MIT
