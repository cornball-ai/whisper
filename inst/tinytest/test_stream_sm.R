# Tests for the streaming turn state machine, LocalAgreement helpers,
# and PCM conversion. All pure R: no torch, no audio, no model.

frame_ms <- whisper:::VAD_FRAME_MS
frames_for <- function(ms) as.integer(ceiling(ms / frame_ms))

run_probs <- function(sm, probs) {
  events <- list()
  for (p in probs) {
    step <- whisper:::vad_sm_step(sm, p)
    sm <- step$sm
    if (!is.null(step$event)) events[[length(events) + 1L]] <- step$event
  }
  list(sm = sm, events = events)
}

# --- onset ------------------------------------------------------------

# Sustained speech opens a turn once min_speech_ms accumulates
sm <- whisper:::vad_sm(min_speech_ms = 250)
res <- run_probs(sm, rep(0.9, frames_for(250)))
expect_equal(length(res$events), 1L)
expect_equal(res$events[[1]]$type, "onset")
expect_equal(res$events[[1]]$start_frame, 1L)
expect_equal(res$sm$state, "in_speech")

# A blip shorter than min_speech_ms never opens one
sm <- whisper:::vad_sm(min_speech_ms = 250)
res <- run_probs(sm, c(rep(0.9, 3), rep(0.0, 30)))
expect_equal(length(res$events), 0L)
expect_equal(res$sm$state, "listening")

# The onset start_frame points at the run start, not the confirm frame
sm <- whisper:::vad_sm(min_speech_ms = 250)
res <- run_probs(sm, c(rep(0.0, 10), rep(0.9, frames_for(250))))
expect_equal(res$events[[1]]$start_frame, 11L)

# Hysteresis: below onset_prob but above offset_prob extends a run
# without being able to start one
sm <- whisper:::vad_sm(onset_prob = 0.5, offset_prob = 0.35,
  min_speech_ms = 250)
res <- run_probs(sm, rep(0.4, 40))
expect_equal(length(res$events), 0L) # 0.4 never starts a run
sm <- whisper:::vad_sm(onset_prob = 0.5, offset_prob = 0.35,
  min_speech_ms = 250)
res <- run_probs(sm, c(0.9, rep(0.4, frames_for(250))))
expect_equal(res$events[[1]]$type, "onset") # but extends one 0.9 started

# --- endpoint ---------------------------------------------------------

# Silence after speech fires an endpoint at the silence target
sm <- whisper:::vad_sm(min_speech_ms = 250, endpoint_silence_ms = 700)
speech <- frames_for(250)
res <- run_probs(sm, c(rep(0.9, speech), rep(0.0, frames_for(700))))
types <- vapply(res$events, function(e) e$type, character(1))
expect_equal(types, c("onset", "endpoint"))
expect_equal(res$events[[2]]$end_frame, speech) # last speech frame
expect_equal(res$sm$state, "listening")

# Silence shorter than the target never endpoints, and speech resets the
# silence counter
sm <- whisper:::vad_sm(min_speech_ms = 250, endpoint_silence_ms = 700)
res <- run_probs(sm, c(rep(0.9, speech), rep(0.0, frames_for(500)),
  rep(0.9, 5), rep(0.0, frames_for(500))))
types <- vapply(res$events, function(e) e$type, character(1))
expect_equal(types, "onset")

# A lowered silence_target_ms (the punctuation-aware path) endpoints
# sooner; the target resets to the base after the endpoint
sm <- whisper:::vad_sm(min_speech_ms = 250, endpoint_silence_ms = 700)
res <- run_probs(sm, rep(0.9, speech))
sm <- res$sm
sm$silence_target_ms <- 500
res <- run_probs(sm, rep(0.0, frames_for(500)))
expect_equal(res$events[[1]]$type, "endpoint")
expect_equal(res$sm$silence_target_ms, 700)

# Two turns through one machine: absolute frame numbering carries over
sm <- whisper:::vad_sm(min_speech_ms = 250, endpoint_silence_ms = 700)
turn <- c(rep(0.9, speech), rep(0.0, frames_for(700)))
res <- run_probs(sm, c(turn, turn))
types <- vapply(res$events, function(e) e$type, character(1))
expect_equal(types, c("onset", "endpoint", "onset", "endpoint"))
expect_equal(res$events[[3]]$start_frame, length(turn) + 1L)
expect_equal(res$events[[4]]$end_frame, length(turn) + speech)

# --- LocalAgreement helpers -------------------------------------------

expect_equal(whisper:::token_lcp(integer(0), integer(0)), 0L)
expect_equal(whisper:::token_lcp(1:5, integer(0)), 0L)
expect_equal(whisper:::token_lcp(1:5, 1:5), 5L)
expect_equal(whisper:::token_lcp(1:5, 1:3), 3L)
expect_equal(whisper:::token_lcp(c(1L, 2L, 9L), c(1L, 2L, 3L, 4L)), 2L)
expect_equal(whisper:::token_lcp(c(9L, 2L), c(1L, 2L)), 0L)

expect_true(whisper:::ends_sentence("So it goes."))
expect_true(whisper:::ends_sentence("Are you there?"))
expect_true(whisper:::ends_sentence("Stop!"))
expect_true(whisper:::ends_sentence("Well then..."))
expect_true(whisper:::ends_sentence("¿Donde? "))
expect_true(whisper:::ends_sentence("He said \"done.\""))
expect_false(whisper:::ends_sentence("and then we"))
expect_false(whisper:::ends_sentence("the U.S. team went to"))
expect_false(whisper:::ends_sentence(""))

# --- PCM conversion ---------------------------------------------------

# int16 values scale to [-1, 1)
expect_equal(whisper:::.pcm_to_float(c(0L, 16384L, -32768L)),
  c(0, 0.5, -1))

# raw S16LE round-trips through writeBin
x <- c(0L, 1000L, -1000L, 32767L, -32768L)
bytes <- writeBin(x, raw(), size = 2L, endian = "little")
expect_equal(whisper:::.pcm_to_float(bytes), x / 32768)

# floats pass through untouched
expect_equal(whisper:::.pcm_to_float(c(-0.5, 0.25)), c(-0.5, 0.25))

# odd byte counts are an error, as is anything non-numeric
expect_error(whisper:::.pcm_to_float(as.raw(c(1, 2, 3))))
expect_error(whisper:::.pcm_to_float("audio"))
