# stream_decode.R
# Incremental transcription for whisper_stream(): LocalAgreement-2
# (Machacek et al. 2023) over repeated decodes of the growing utterance
# buffer, with the committed tokens forced as the decoder prefix so no
# later decode can contradict text already emitted as stable.
#
# Event semantics (the gpu_voice.proto contract): stable text APPENDS --
# the turn's text is the exact concatenation of stable events, joining
# whitespace included -- and provisional text REPLACES, each event being
# the whole uncommitted tail. Committing at token level is what makes the
# append exact: text is only ever decoded over a committed token range,
# never re-tokenized.

# Longest common prefix of two integer token vectors (pure; unit-tested).
token_lcp <- function(a, b) {
  n <- min(length(a), length(b))
  if (n == 0L) {
    return(0L)
  }
  neq <- which(a[seq_len(n)] != b[seq_len(n)])
  if (length(neq) == 0L) n else neq[1L] - 1L
}

# Does this transcript tail read as a finished sentence? Decides the
# adaptive silence target (punctuation-aware endpointing).
ends_sentence <- function(text) {
  grepl("[.?!…。？！]['\")”’]?\\s*$", text)
}

# Per-turn decoder state. `committed` are content token ids already
# emitted as stable; `prev_tail` is the previous decode's uncommitted
# content, the other half of the LocalAgreement-2 comparison.
stream_decoder <- function(language = NULL, task = "transcribe") {
  list(
    language = language,
    task = task,
    committed = integer(0),
    prev_tail = NULL
  )
}

stream_decoder_reset_turn <- function(sd) {
  sd$committed <- integer(0)
  sd$prev_tail <- NULL
  sd
}

# One decode over the utterance buffer (float samples at 16 kHz).
# final = FALSE: a LocalAgreement tick -- greedy, temperature 0 only.
# final = TRUE: the flush decode -- allowed the quality ladder, commits
# everything.
# Returns list(sd, events, text) where text is the turn text so far
# (committed + tail) for the punctuation-aware silence target.
stream_decode_tick <- function(
  sd,
  pipe,
  samples,
  final = FALSE,
  final_temperatures = c(0, 0.2, 0.4, 0.6, 0.8, 1.0),
  final_beam_size = 1L,
  no_speech_threshold = 0.6,
  logprob_threshold = -1.0
) {
  config <- pipe$config
  tokenizer <- pipe$tokenizer
  special <- whisper_special_tokens(config$model_name)

  mel <- audio_to_mel(samples, n_mels = config$n_mels,
    device = pipe$device, dtype = pipe$dtype)

  # Detect language once per stream, on the first decode; pinned after.
  if (is.null(sd$language)) {
    detection <- detect_language_from_mel(pipe$model, mel, config,
      pipe$device)
    sd$language <- detection$language
  }

  initial <- get_initial_tokens(sd$language, sd$task,
    model = config$model_name, timestamps = FALSE)
  start_tokens <- c(initial, sd$committed)
  tokens <- torch::torch_tensor(matrix(start_tokens, nrow = 1L),
    dtype = torch::torch_long(), device = pipe$device)

  torch::with_no_grad({
    encoder_output <- pipe$model$encode(mel)
  })

  decode_result <- decode_with_fallback(pipe$model, encoder_output,
    tokens, tokenizer,
    temperatures = if (final) final_temperatures else 0,
    beam_size = if (final) final_beam_size else 1L,
    best_of = 1L,
    max_length = config$n_text_ctx %/% 2L + length(start_tokens),
    timestamps = FALSE,
    no_speech_threshold = no_speech_threshold,
    logprob_threshold = logprob_threshold,
    device = pipe$device)

  generated <- decode_result$tokens
  content <- generated[seq_along(generated) > length(start_tokens)]
  # Timestamps are disabled and specials suppressed, but never let a
  # stray non-text token into the committed prefix.
  content <- content[content < special$timestamp_begin]

  # A buffer that reads as silence (the batch path's no-speech gate)
  # contributes nothing; keep the previous tail rather than "agreeing"
  # with a hallucination-prone empty decode.
  is_silence <- !is.na(decode_result$no_speech_prob) &&
    decode_result$no_speech_prob > no_speech_threshold &&
    !is.na(decode_result$avg_logprob) &&
    decode_result$avg_logprob < logprob_threshold
  if (is_silence) {
    content <- integer(0)
  }

  events <- list()

  n_commit <- if (final) {
    length(content)
  } else if (is.null(sd$prev_tail)) {
    0L # first hypothesis of the turn: nothing to agree with yet
  } else {
    token_lcp(sd$prev_tail, content)
  }

  if (n_commit > 0L) {
    stable_text <- tokenizer$decode(content[seq_len(n_commit)])
    sd$committed <- c(sd$committed, content[seq_len(n_commit)])
    events[[length(events) + 1L]] <- list(type = "transcript",
      text = stable_text, stable = TRUE)
  }

  tail_tokens <- content[seq_along(content) > n_commit]
  if (!final && length(tail_tokens) > 0L) {
    events[[length(events) + 1L]] <- list(type = "transcript",
      text = tokenizer$decode(tail_tokens), stable = FALSE)
  }
  sd$prev_tail <- if (is_silence) sd$prev_tail else tail_tokens

  text <- if (length(c(sd$committed, tail_tokens)) > 0L) {
    tokenizer$decode(c(sd$committed, tail_tokens))
  } else {
    ""
  }

  list(sd = sd, events = events, text = text)
}
