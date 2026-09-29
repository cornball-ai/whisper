# stream_decode.R
# Incremental transcription for whisper_stream(): LocalAgreement-2
# (Machacek et al. 2023) over repeated decodes of the growing utterance
# buffer, with the committed tokens forced as the decoder prefix so no
# later decode can contradict text already emitted as stable.
#
# Event semantics (the gpu_voice.proto contract): stable text APPENDS --
# the turn's text is the exact concatenation of stable events, joining
# whitespace included -- and provisional text REPLACES, each event being
# the whole uncommitted tail (an empty provisional clears a stale one).
#
# Commitment is tracked at token level, but event text is NEVER produced
# by decoding a token slice on its own: byte-level BPE can split a UTF-8
# character across two tokens, and the tokenizer drops incomplete
# sequences, so slice-decoding loses characters at commit boundaries
# (tiny tokenizes "¢" as two tokens; each half decodes to "").
# Instead every event is a character delta between decodes of full token
# prefixes, where partial characters at the boundary simply stay pending
# until the tokens completing them commit.

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

# The silence target for the current tick (pure; unit-tested). Terminal
# punctuation shortens the timer only once the turn holds at least
# punct_min_speech_ms of actual speech: on a shorter turn the transcript
# is the least trustworthy evidence in the stream (a garbled 400 ms
# fragment reading " Not!" is how JFK's opening lost its first words),
# so an apparent sentence end there falls back to the no-transcript
# target instead of being given authority over the endpoint.
silence_target <- function(
  text,
  turn_speech_ms,
  endpoint_silence_ms,
  punct_silence_ms,
  midsentence_silence_ms,
  punct_min_speech_ms
) {
  if (!nzchar(text)) {
    endpoint_silence_ms
  } else if (ends_sentence(text)) {
    if (turn_speech_ms >= punct_min_speech_ms) {
      punct_silence_ms
    } else {
      endpoint_silence_ms
    }
  } else {
    midsentence_silence_ms
  }
}

# Per-turn decoder state. `committed` are content token ids already
# forced as the decoder prefix; `prev_tail` is the previous decode's
# uncommitted content (the other half of the LocalAgreement-2
# comparison; NULL means no hypothesis yet). `stable_sent` and
# `provisional_sent` are the exact strings the consumer has, the
# reference points for the delta/replace emission below.
stream_decoder <- function(language = NULL, task = "transcribe") {
  list(
    language = language,
    task = task,
    committed = integer(0),
    prev_tail = NULL,
    stable_sent = "",
    provisional_sent = ""
  )
}

stream_decoder_reset_turn <- function(sd) {
  sd$committed <- integer(0)
  sd$prev_tail <- NULL
  sd$stable_sent <- ""
  sd$provisional_sent <- ""
  sd
}

# Advance the LocalAgreement state with a fresh hypothesis. Pure given
# decode_fn (integer token ids -> text), so it unit-tests with a stub
# tokenizer. `content` is the hypothesis's content tokens beyond the
# committed prefix. Returns list(sd, events, text).
la_advance <- function(sd, content, final, decode_fn) {
  events <- list()

  n_commit <- if (final) {
    length(content)
  } else if (is.null(sd$prev_tail)) {
    0L # first hypothesis of the turn: nothing to agree with yet
  } else {
    token_lcp(sd$prev_tail, content)
  }

  if (n_commit > 0L) {
    sd$committed <- c(sd$committed, content[seq_len(n_commit)])
    # Delta against the full committed prefix, never a slice decode. A
    # trailing partial character decodes to nothing now and surfaces in a
    # later delta, once the tokens completing it commit.
    full_stable <- decode_fn(sd$committed)
    if (startsWith(full_stable, sd$stable_sent)) {
      delta <- substring(full_stable, nchar(sd$stable_sent) + 1L)
      if (nzchar(delta)) {
        events[[length(events) + 1L]] <- list(type = "transcript",
          text = delta, stable = TRUE)
        sd$stable_sent <- full_stable
      }
    }
    # A non-prefix full_stable cannot happen with byte-level BPE (later
    # decodes only ever complete the dropped trailing bytes); if it ever
    # did, emitting would restate or contradict, so we hold position.
  }

  tail_tokens <- content[seq_along(content) > n_commit]
  full_text <- if (length(c(sd$committed, tail_tokens)) > 0L) {
    decode_fn(c(sd$committed, tail_tokens))
  } else {
    ""
  }

  if (!final) {
    provisional <- if (startsWith(full_text, sd$stable_sent)) {
      substring(full_text, nchar(sd$stable_sent) + 1L)
    } else {
      ""
    }
    # REPLACE semantics: emit on every change, the empty string included
    # -- that is how a stale provisional is cleared when a new hypothesis
    # retracts it.
    if (!identical(provisional, sd$provisional_sent)) {
      events[[length(events) + 1L]] <- list(type = "transcript",
        text = provisional, stable = FALSE)
      sd$provisional_sent <- provisional
    }
  }
  sd$prev_tail <- tail_tokens

  list(sd = sd, events = events, text = full_text)
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

  # First-token blank/EOT suppression belongs to the original prompt
  # boundary only: once committed tokens extend the prefix, the decode
  # must be free to terminate immediately (the utterance may already be
  # fully committed).
  decode_result <- decode_with_fallback(pipe$model, encoder_output,
    tokens, tokenizer,
    temperatures = if (final) final_temperatures else 0,
    beam_size = if (final) final_beam_size else 1L,
    best_of = 1L,
    max_length = config$n_text_ctx %/% 2L + length(start_tokens),
    timestamps = FALSE,
    no_speech_threshold = no_speech_threshold,
    logprob_threshold = logprob_threshold,
    suppress_blank = length(sd$committed) == 0L,
    device = pipe$device)

  generated <- decode_result$tokens
  content <- generated[seq_along(generated) > length(start_tokens)]
  # Timestamps are disabled and specials suppressed, but never let a
  # stray non-text token into the committed prefix.
  content <- content[content < special$timestamp_begin]

  # A buffer that reads as silence (the batch path's no-speech gate)
  # contributes an empty hypothesis: nothing commits, and a stale
  # provisional is retracted rather than left standing.
  is_silence <- !is.na(decode_result$no_speech_prob) &&
    decode_result$no_speech_prob > no_speech_threshold &&
    !is.na(decode_result$avg_logprob) &&
    decode_result$avg_logprob < logprob_threshold
  if (is_silence) {
    content <- integer(0)
  }

  la_advance(sd, content, final, tokenizer$decode)
}
