# whisper

Native R torch implementation of OpenAI Whisper for speech-to-text transcription.

## Architecture

```
Audio (WAV/MP3) -> Mel Spectrogram -> Encoder (transformer) -> Decoder (cross-attn) -> Tokens -> Text
```

## Model Sizes

| Model | Layers | d_model | Heads | FFN | Params | Mel bins |
|-------|--------|---------|-------|-----|--------|----------|
| tiny | 4 | 384 | 6 | 1536 | 39M | 80 |
| base | 6 | 512 | 8 | 2048 | 74M | 80 |
| small | 12 | 768 | 12 | 3072 | 244M | 80 |
| medium | 24 | 1024 | 16 | 4096 | 769M | 80 |
| large-v3 | 32 | 1280 | 20 | 5120 | 1550M | 128 |

## Key Exports

- `transcribe(file, model, language, timestamps, word_timestamps, beam_size, temperatures)` - Main transcription function
- `whisper_pipeline(model)` - Load model once, call `$transcribe()` repeatedly
- `load_whisper_model(model, device, dtype)` - Load model weights
- `audio_to_mel(file, n_mels)` - Convert audio to mel spectrogram
- `whisper_tokenizer()` - Get BPE tokenizer

## Usage

```r
library(whisper)

# Transcribe audio
result <- transcribe("audio.wav", model = "tiny")
result$text

# Segment timestamps (uses Whisper's built-in timestamp tokens)
result <- transcribe("audio.wav", timestamps = TRUE)
result$segments  # data.frame(start, end, text)

# Word-level timestamps (cross-attention DTW alignment)
result <- transcribe("audio.wav", word_timestamps = TRUE)
result$words  # data.frame(word, start, end)

# Beam search (better accuracy, slower)
result <- transcribe("audio.wav", beam_size = 5L)

# Temperature fallback (handles difficult audio)
result <- transcribe("audio.wav", beam_size = 5L,
                     temperatures = c(0, 0.2, 0.4, 0.6, 0.8, 1.0))
```

## Development

```bash
# Build and test
r -e 'tinyrox::document(); tinypkgr::install(); tinytest::test_package("whisper")'

# Quick iteration
r -e 'tinypkgr::load_all(); transcribe("test.wav")'
```

## Weight Loading

Uses safetensors format from HuggingFace:
- `openai/whisper-tiny`
- `openai/whisper-base`
- `openai/whisper-small`
- `openai/whisper-medium`
- `openai/whisper-large-v3`

## File Structure

- `R/transcribe.R` - Main API, greedy/beam/sample decode, timestamp logit rules, temperature fallback
- `R/alignment.R` - DTW alignment, word timestamp computation
- `R/audio.R` - Audio to mel spectrogram
- `R/encoder.R` - Encoder transformer (with `need_weights` dual-path attention)
- `R/decoder.R` - Decoder with cross-attention
- `R/model.R` - Full model + weight loading
- `R/tokenizer.R` - Whisper BPE tokenizer
- `R/config.R` - Model configurations + alignment heads
- `R/download.R` - HuggingFace model download
- `R/devices.R` - Device/dtype management

## Status

**On CRAN** - https://cran.r-project.org/package=whisper

### Features

- Transcription and translation (any language to English)
- All model sizes: tiny, base, small, medium, large-v3
- CPU and CUDA support
- Segment-level timestamps (Whisper timestamp tokens with logit suppression)
- Word-level timestamps (cross-attention DTW alignment)
- Pre-computed mel filterbank from official Whisper
- HuggingFace model downloads via `hfhub`
- KV cache for efficient incremental decoding
- Long audio support (automatic chunking with time offsets)
- Beam search decoding with length-normalized scoring
- Temperature sampling with best-of selection
- Temperature fallback with compression ratio and logprob quality checks

### R torch notes

- Use `as.array()` for tensor to R conversion (R has native array support)
- Embeddings use 1-based indexing (add 1 to 0-based token IDs)
- `torch$argmax()` returns 1-indexed values

### Known Limitations

- Translation quality varies by model size (larger models work better)
