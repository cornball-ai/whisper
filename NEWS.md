# whisper 0.5.1

* `whisper_tune_gc()` no longer initializes CUDA before setting the options
  it exists to set. torch reads the allocator rates exactly once, at CUDA
  init, and the function's own device and dtype resolution
  (`parse_device("auto")`, `parse_dtype()`, and allocating a tensor to
  measure its element size) triggered that init first -- so the options
  landed after they had been read, did nothing for the rest of the session,
  and the usual success message was printed anyway. Device and dtype are
  now resolved from strings and `nvidia-smi`, which reports GPU name and
  memory without creating a CUDA context. This matters most where the
  function is most useful: tuning once at process start, before several
  models load.

# whisper 0.5.0

* In-process model residency: keep a model's weights as page-locked (pinned)
  CPU tensors and create/destroy its GPU representation on demand, so
  switching models on a small GPU is a sub-second DMA copy instead of a
  full reload from disk.

  ```r
  res <- resident_load("medium")   # loads, pins weights in host RAM
  resident_activate(res)           # DMA copy to GPU: ~0.25 s for 1.4 GB
  resident_transcribe(res, "audio.mp3", timestamps = TRUE)
  resident_deactivate(res)         # VRAM freed; weights stay pinned in RAM
  resident_activate(res)           # fast again -- no disk involved
  resident_unload(res)
  ```

  Transitions are transactional: a partially-failed activation (e.g. GPU
  out-of-memory) rolls back to the pinned host state and verifies it; an
  unverifiable rollback fail-closes the handle. `resident_status()` reports
  state, per-tensor logical byte counts, and a content identity (weights
  sha256, HF repo and snapshot revision, resolved dtype). New functions:
  `resident_load()`, `resident_activate()`, `resident_deactivate()`,
  `resident_transcribe()`, `resident_status()`, `resident_unload()`.

  `resident_deactivate(release = )` chooses who gets the freed VRAM, and
  on a small card it is worth ~10x. The default `TRUE` returns the CUDA
  allocator's blocks to the driver, so other processes see the memory
  free; the next activation then re-acquires every block from the driver
  (medium fp32 on a 6 GB card: 948 tensors, 2.85 GB, 9.2 s / 0.31 GB/s).
  `FALSE` keeps the blocks pooled for the next model in the same process
  to reuse: the identical activation takes 0.86 s / 3.29 GB/s, which is
  the card's raw pinned-DMA bandwidth. Weights are freed and `gpu_bytes`
  reaches zero either way; only the pool differs.

* whisper now requires R >= 4.5.0 (for `tools::sha256sum()`).

# whisper 0.4.1

* `transcribe()` results now carry the shape subtitle tooling expects, so they
  feed `subtitles::whisper_to_srt()` and `subtitles::whisper_to_ass()`
  directly:

  ```r
  x <- whisper::transcribe("video.mp4", timestamps = TRUE)
  subtitles::whisper_to_srt(x, "video.srt")
  ```

  A result with segments gains a `data` frame of `from`/`to` timestamp strings
  and `text`, and class `c("whisper_result", "whisper_transcription")`. The
  change is additive: `text`, `segments`, and `words` are unchanged, and
  results without segments (`timestamps = FALSE`) are returned as before. Word
  timings still require `word_timestamps = TRUE`, which
  `whisper_to_ass(karaoke = TRUE)` needs.

# whisper 0.4.0

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
* Silence handling now matches the reference Whisper, fixing transcripts that
  ran past the end of short audio. Three changes, ported from
  `openai-whisper`: decoding suppresses non-speech tokens (brackets, music
  notes, speaker tags) and control tokens at every step, so output no longer
  contains `[BLANK_AUDIO]`/`[MUSIC PLAYING]`-style annotations; the seek loop
  decodes only the real audio (`content_frames`), not the fixed 30s of mel
  padding, so a 7s clip no longer trails off into hallucinated text up to 30s;
  and a no-speech-probability gate skips windows that read as silence. The
  special-token table gains `sot_lm` and `sot_prev`.
* Bound and mitigate degenerate repetition loops, matching the reference. A
  long non-speech sound (e.g. a laugh) could make the decoder emit one token
  ("ha") hundreds of times - garbage output, and enough accumulated
  cross-attention to exhaust memory on a small GPU. Decoding is now capped at
  half the text context (the reference's `sample_len`) rather than the full
  context, and the default `temperatures` enable the existing compression-ratio
  fallback, which re-decodes too-repetitive output at a higher temperature.
* Fix `tokenizer_encode()` crashing for models whose `vocab.json` omits the
  `<|endoftext|>` key (large-v3): the end-of-text id now comes from the
  special-token table (as in the Python reference, which keeps special tokens
  out of the BPE vocab), and the lookup can no longer return a list. A
  regression test covers a vocab without the key, and `encode_special()`
  resolves the core special tokens from the table too.
* `whisper_dtype()` now falls back to float32 on the GTX 16-series
  (TU116/TU117: GTX 1630/1650/1660 and Ti/Super variants), which compute fp16
  incorrectly and return NaN (seen as repeated "!" tokens). Detection is by GPU
  name, CUDA-gated and tryCatch-guarded (dormant on non-CUDA/CRAN machines);
  pass `dtype = "float16"` to override.
* New `whisper_tune_gc()`: opt-in helper that tunes torch's CUDA allocator GC
  rates for inference. No-op off CUDA, and only sets options that are unset.
* Scaled dot-product attention now calls the exported
  `torch::torch_scaled_dot_product_attention()` instead of reaching into
  torch's namespace; the torch dependency is floored at 0.17.0, where it is
  exported.
* README performance table refreshed for the JIT word-timestamp path.

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
