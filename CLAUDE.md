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

# With timestamps
result <- transcribe("audio.wav", model = "small", timestamps = TRUE)
print(result$segments)
```

## Development

```bash
# Build and test
r -e 'rhydrogen::document(); rhydrogen::install(); tinytest::test_package("whisper")'

# Quick iteration
r -e 'rhydrogen::load_all(); transcribe("test.wav")'
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

**Work in Progress** - Core architecture is complete but transcription output needs validation.

### Completed

- Package structure with tinyverse conventions
- Encoder (conv stem + transformer) with sinusoidal positional encoding
- Decoder (self-attention + cross-attention) with learned positional embedding
- Weight loading from HuggingFace safetensors
- KV cache for incremental decoding
- Basic tokenizer with BPE

### Known Issues

1. **Transcription quality**: Model loads and runs but output doesn't match Python Whisper. Likely due to:
   - Mel filterbank differences (computing on-the-fly vs. Whisper's pre-computed)
   - STFT implementation details
   - Numerical precision differences

2. **R torch quirks**:
   - Use `as.array()` instead of `$numpy()` for tensor to R conversion
   - Embeddings use 1-based indexing (add 1 to 0-based token IDs)

### Validation TODO

1. Compare mel spectrogram output with Python Whisper on same audio
2. Compare encoder output on same mel input
3. Compare decoder output on same encoder states
4. Identify numerical divergence point

### Suggested Improvements

- Use `hfhub` package for model downloads instead of manual download.file()
- Load pre-computed mel filterbank from Whisper's mel_filters.npz
- Add beam search decoding option
