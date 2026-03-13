## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* local Ubuntu 24.04, R 4.5.2
* GitHub Actions (r-ci): ubuntu-latest, macos-latest
* win-builder R-devel

## Changes since last CRAN release (0.1.0)

- `whisper_pipeline()` for cached model reuse across multiple transcriptions
- SDPA attention (FlashAttention on GPU)
- Segment-level and word-level timestamps via DTW alignment
- Beam search decoding with temperature sampling and fallback
- Automatic language detection (`detect_language()`, `language = NULL` default)
- Hardcoded special token table (eliminates `added_tokens.json` download)
- Fixed invalid multibyte string crash in BPE decoder
- Fixed DTW boundary guards and seek loop in `transcribe_chunk()`
- Fixed win-builder test crash: added `torch_is_installed()` guard

## Notes

This package uses `torch_scaled_dot_product_attention()` from torch,
which is not yet exported in the current CRAN release. We access it via
`get("torch_scaled_dot_product_attention", envir = asNamespace("torch"))`.
The function will be exported in the next torch release
(PR: https://github.com/mlverse/torch/pull/1404).

This package provides a native R implementation of OpenAI's Whisper
speech-to-text model. Model weights are downloaded from HuggingFace
on first use (ranging from 145MB for tiny to 3GB for large-v3).
