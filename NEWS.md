# whisper 0.3.0

* Language auto-detection: `transcribe()` now defaults to `language = NULL`,
  which detects the spoken language from the audio before decoding. New
  exported function `detect_language()` for standalone language identification.
  **Breaking**: previous default was `language = "en"`. Code relying on the
  default now auto-detects instead of assuming English. Pass `language = "en"`
  explicitly to restore old behavior.
* Segment-level and word-level timestamps via DTW alignment
* Beam search decoding with temperature sampling and fallback
* SDPA attention (FlashAttention on GPU)
* `whisper_pipeline()` for cached model reuse across multiple transcriptions
* Hardcoded special token table (eliminates `added_tokens.json` download)
* Fixed invalid multibyte string crash in BPE decoder
* Fixed DTW boundary guards and seek loop in `transcribe_chunk()`

# whisper 0.1.0

* Initial CRAN submission
* Native R torch implementation of OpenAI Whisper
* Support for all model sizes: tiny, base, small, medium, large-v3
* Automatic model download from HuggingFace
* Model-specific special token handling for large-v3 compatibility
* KV caching for efficient autoregressive decoding
* Long audio chunking for files longer than 30 seconds
* Optional timestamp and segment extraction
