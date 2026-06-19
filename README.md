# whisper

Native R torch implementation of OpenAI Whisper for speech-to-text transcription.

## Installation

```r
install.packages("whisper")
torch::install_torch()  # one-time: downloads the C++ backend
```

Or install the development version from GitHub:

```r
remotes::install_github("cornball-ai/whisper")
torch::install_torch()
```

`{whisper}` uses `{torch}` for inference. After installing the R package, `install_torch()` downloads the compiled C++ libraries (Lantern). You only need to run it once.

**littler/r2u users:** If `install_torch()` fails with a permissions error, torch was installed to the system library. Use sudo for the one-time Lantern download:

```bash
sudo r -e 'torch::install_torch()'
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

## Serving

`serve()` runs an OpenAI-compatible STT server that loads the model once and
keeps it resident, so a client (or [`{stt.api}`](https://github.com/cornball-ai/stt.api))
can transcribe over HTTP:

```r
whisper::serve(model = "small", port = 7809L)
```

```bash
curl -X POST http://localhost:7809/v1/audio/transcriptions \
  -F file=@speech.wav -F response_format=verbose_json
```

Endpoints: `GET /health`, `POST /v1/audio/transcriptions`, `POST /v1/audio/translations`.
It's built on base R sockets (no extra dependencies). A systemd unit ships in
`system.file("whisper.service", package = "whisper")`.

## Performance and hardware notes

- **JIT decode (CUDA).** Each token's decoder forward runs as a single
  TorchScript call (`jit = TRUE`, the default on CUDA) instead of dozens of
  per-op R calls, several times faster than the eager path and equivalent
  token-for-token. Covers greedy and word-timestamp runs; beam search and CPU
  use the eager decoder.
- **GTX 16-series fp16 is broken.** The GTX 1630/1650/1660 (TU116/TU117) compute
  fp16 incorrectly and return NaN (transcription comes out as repeated `!`).
  `whisper_dtype()` auto-falls back to float32 on those cards; pass
  `dtype = "float16"` to override.
- **`whisper_tune_gc()`** is an opt-in helper that tunes torch's CUDA allocator
  GC. It is largely inert for whisper (whisper is dispatch-bound, not
  GC-bound — JIT is the lever); it's kept as cheap insurance.

## Models

| Model | Parameters | Disk (fp32) | English WER | Peak VRAM (CUDA fp16) | Speed* |
|-------|------------|-------------|-------------|----------------------|--------|
| tiny | 39M | 151 MB | ~9% | 564 MiB | 1.1s |
| base | 74M | 290 MB | ~7% | 734 MiB | 1.1s |
| small | 244M | 967 MB | ~5% | 1,454 MiB | 1.2s |
| medium | 769M | 3.0 GB | ~4% | 3,580 MiB | 1.3s |
| large-v3 | 1550M | 6.2 GB | ~3% | 3,892 MiB | 2.7s |

*Speed is a warm transcribe of a 17s clip on an RTX 5060 Ti with
`word_timestamps = TRUE` (the heavier path; plain greedy is several times
faster, e.g. large-v3 ~1.4s), excluding one-time model load and JIT
compilation. Peak VRAM includes ~364 MiB torch CUDA context overhead.

Models are downloaded from HuggingFace and cached in `~/.cache/huggingface/` unless otherwise specified.

## License

MIT
