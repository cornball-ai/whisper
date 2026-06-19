## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* local Ubuntu 24.04, R 4.6.0
* GitHub Actions (r-ci): ubuntu-latest, macos-latest
* Windows 10: R 4.6.0 and R-devel
* win-builder: R-release and R-devel

## Changes since last CRAN release (0.3.0)

- Silence handling now matches the reference Whisper: decoding suppresses
  non-speech and control tokens, the seek loop decodes only the real audio
  (not the fixed 30s of mel padding), and a no-speech-probability gate skips
  silent windows. This fixes transcripts of short audio that ran past the end
  of speech with bracketed annotations or hallucinated text.
- New `serve()`: an OpenAI-compatible HTTP speech-to-text server built on
  base R sockets (no new dependencies); loads the model once and stays
  resident.
- JIT decoding on CUDA via `torch::jit_compile` (new `jit` argument,
  default TRUE on CUDA), token-for-token equivalent to the eager path and
  several times faster. Covers greedy and word-timestamp decoding.
- New `whisper_tune_gc()`: opt-in helper for torch's CUDA allocator GC
  rates. It is a no-op off CUDA and only sets options that are not already
  set, so it has no effect during checks on CUDA-less machines.
- `whisper_dtype()` falls back to float32 on the GTX 16-series, whose fp16
  is hardware-broken; CUDA-gated, so no effect during checks.
- Degenerate repetition loops are now bounded (decoding capped at half the
  text context) with a compression-ratio temperature fallback, matching the
  reference; previously a long non-speech sound could loop a single token.
- Fixed `tokenizer_encode()` crashing on models whose `vocab.json` omits the
  `<|endoftext|>` key (large-v3); the end-of-text id now comes from the
  special-token table, with a regression test, and `encode_special()` resolves
  the core special tokens from that table too.
- Scaled dot-product attention now calls the exported
  `torch::torch_scaled_dot_product_attention()` (torch >= 0.17.0).

## Notes

This package provides a native R implementation of OpenAI's Whisper
speech-to-text model. Model weights are downloaded from HuggingFace on
first use (145MB for tiny to 3GB for large-v3). GPU/CUDA-specific code
paths (JIT decode, GC tuning, the fp16 fallback, `serve()`) are gated and
do not run during checks on machines without CUDA.
