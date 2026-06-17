# whisper 0.3.0.2 (development)

* New `serve()`: a single-process HTTP server exposing the model over an
  OpenAI-compatible `POST /v1/audio/transcriptions` (and `/translations`)
  endpoint, plus `GET /health`. Built on base R sockets (no new
  dependencies), it loads the model once and keeps it resident, so it
  drops in for the OpenAI API or a Whisper container - point `stt.api` at
  it with `set_stt_base()`. Multipart/form-data uploads are parsed in
  base R; responses are `text`, `json`, or `verbose_json` (with segment
  timings). On CUDA it tunes the allocator GC and uses the TorchScript
  decode step. An example systemd unit ships in
  `system.file("whisper.service", package = "whisper")`, set up to run as
  a second always-on process alongside a chatterbox TTS server.

# whisper 0.3.0.1 (development)

* Greedy decoding on CUDA now runs each token's decoder forward (all
  layers' self-attention, cross-attention, and FFN, plus the final
  LayerNorm) as one `jit_compile`'d TorchScript call, instead of dozens
  of dispatched R->torch calls per token. Same motivation as chatterbox's
  `t3_inference_jit`: even lean eager R hits a per-op dispatch floor, and
  collapsing the per-token forward into one libtorch call removes it
  without compiled code. Token-for-token equivalent to the eager path
  (verified in `test_decode_jit.R`); ~2.5x faster end-to-end on `medium`
  for a short clip, more on longer transcripts. On by default via the new
  `jit` argument to `transcribe()`/`whisper_pipeline()`; pass `jit = FALSE`
  to force the eager decoder. No effect on CPU, beam search, or
  word-timestamp runs, which use the eager path.
* New `whisper_tune_gc()`: opt-in helper that raises torch's CUDA
  allocator GC floor (`torch.cuda_allocator_reserved_rate`) to the model's
  footprint as a fraction of VRAM and lifts `torch.threshold_call_gc` off
  its default, so GC stops firing on nearly every allocation during
  inference. Call it before `load_whisper_model()`. No-op off CUDA and
  only sets options that are not already set. See torch's memory-management
  vignette.

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
