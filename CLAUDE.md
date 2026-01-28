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

- `transcribe(file, model, language)` - Main transcription function
- `load_whisper_model(model, device, dtype)` - Load model weights
- `audio_to_mel(file, n_mels)` - Convert audio to mel spectrogram
- `whisper_tokenizer()` - Get BPE tokenizer

## Usage

```r
library(whisper)

# Transcribe audio
result <- transcribe("audio.wav", model = "tiny")
print(result$text)
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

- `R/transcribe.R` - Main API
- `R/audio.R` - Audio to mel spectrogram
- `R/encoder.R` - Encoder transformer
- `R/decoder.R` - Decoder with cross-attention
- `R/model.R` - Full model + weight loading
- `R/tokenizer.R` - Whisper BPE tokenizer
- `R/config.R` - Model configurations
- `R/download.R` - HuggingFace model download
- `R/devices.R` - Device/dtype management

## Status

**Ready for CRAN** - Core functionality complete and tested.

### Features

- Transcription and translation (any language to English)
- All model sizes: tiny, base, small, medium, large-v3
- CPU and CUDA support
- Pre-computed mel filterbank from official Whisper
- HuggingFace model downloads via `hfhub`
- KV cache for efficient incremental decoding
- Long audio support (automatic chunking)

### R torch notes

- Use `as.array()` for tensor to R conversion (R has native array support)
- Embeddings use 1-based indexing (add 1 to 0-based token IDs)
- `torch$argmax()` returns 1-indexed values

### Known Limitations

- UTF-8 encoding issues with some non-ASCII characters in output
- Translation quality varies by model size (larger models work better)
- No beam search (greedy decoding only)

### Potential Improvements

- Beam search decoding
- Word-level timestamps (requires cross-attention analysis)
- Fix UTF-8 byte decoding in tokenizer
