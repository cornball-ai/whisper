# whisper 0.3.0.5

* Silence handling now matches the reference Whisper, fixing transcripts that
  ran past the end of short audio. Three changes, ported from
  `openai-whisper`: decoding suppresses non-speech tokens (brackets, music
  notes, speaker tags) and control tokens at every step, so output no longer
  contains `[BLANK_AUDIO]`/`[MUSIC PLAYING]`-style annotations; the seek loop
  decodes only the real audio (`content_frames`), not the fixed 30s of mel
  padding, so a 7s clip no longer trails off into hallucinated text up to 30s;
  and a no-speech-probability gate skips windows that read as silence. The
  special-token table gains `sot_lm` and `sot_prev`.
* New `serve()`: a single-process, OpenAI-compatible HTTP STT server
  (`POST /v1/audio/transcriptions` and `/translations`, `GET /health`) built
  on base R sockets, with no new dependencies. It loads the model once and
  keeps it resident, so it drops in for the OpenAI API or a Whisper container;
  point `stt.api` at it with `set_stt_base()`. Returns `text`, `json`, or
  `verbose_json` (segment timestamps, plus per-word timestamps when the request
  includes `timestamp_granularities[]=word`). An example systemd unit ships in
  `system.file("whisper.service", package = "whisper")`.
* JIT decoding on CUDA: each generated token's decoder forward runs as one
  `jit_compile`'d TorchScript call instead of dozens of dispatched R->torch
  calls, several times faster end-to-end and token-for-token equivalent to the
  eager path. Covers both greedy and word-timestamp decoding. On by default via
  the new `jit` argument to `transcribe()`/`whisper_pipeline()`; pass
  `jit = FALSE` for the eager decoder. No effect on CPU or beam search.
* New `whisper_tune_gc()`: opt-in helper that tunes torch's CUDA allocator GC
  rates for inference. No-op off CUDA, and only sets options that are unset.
* `whisper_dtype()` now falls back to float32 on the GTX 16-series
  (TU116/TU117: GTX 1630/1650/1660 and Ti/Super variants), which compute fp16
  incorrectly and return NaN (seen as repeated "!" tokens). Detection is by GPU
  name, CUDA-gated and tryCatch-guarded (dormant on non-CUDA/CRAN machines);
  pass `dtype = "float16"` to override.

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
