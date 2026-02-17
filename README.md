# whisper

Native R torch implementation of OpenAI Whisper for speech-to-text transcription.

## Installation

```r
install.packages("whisper")
```

Or install the development version from GitHub:

```r
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

# Force CPU (useful if CUDA has issues)
result <- transcribe("audio.wav", device = "cpu")

# Non-English audio (specify language for better accuracy)
allende <- system.file("audio", "allende.mp3", package = "whisper")
result <- transcribe(allende, language = "es")

# Translate to English (quality is model-dependent; larger models work better)
result <- transcribe(allende, task = "translate", language = "es", model = "small")
```

## Timestamps

```r
# Segment-level timestamps
result <- transcribe("audio.wav", timestamps = TRUE)
result$segments
#>   start  end                         text
#> 1  0.00 7.44 Ask not what your country...

# Word-level timestamps (via cross-attention DTW alignment)
result <- transcribe("audio.wav", word_timestamps = TRUE)
result$words
#>      word start  end
#> 1     Ask  0.00 0.54
#> 2     not  0.54 1.16
#> 3    what  1.16 2.46
#> ...
```

Both work with the pipeline API for repeated transcription:

```r
pipe <- whisper_pipeline("tiny")
result <- pipe$transcribe("audio.wav", word_timestamps = TRUE)
result$words
```

## Models

| Model | Parameters | Disk (fp32) | English WER | Peak VRAM (CUDA fp16) | Speed* |
|-------|------------|-------------|-------------|----------------------|--------|
| tiny | 39M | 151 MB | ~9% | 564 MiB | 5.5s |
| base | 74M | 290 MB | ~7% | 734 MiB | 1.9s |
| small | 244M | 967 MB | ~5% | 1,454 MiB | 3.6s |
| medium | 769M | 3.0 GB | ~4% | 3,580 MiB | 8.6s |
| large-v3 | 1550M | 6.2 GB | ~3% | 3,892 MiB | 16.7s |

*Speed measured on RTX 5060 Ti transcribing a 17s audio clip with `word_timestamps = TRUE`.
Peak VRAM includes ~364 MiB torch CUDA context overhead.

Models are downloaded from HuggingFace and cached in `~/.cache/huggingface/` unless otherwise specified.

## License

MIT
