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

# --- LocalAgreement event generation (la_advance) ---------------------
#
# Stub byte-level tokenizer, faithful to decode_bpe_bytes(): tokens map
# to raw bytes, decode concatenates and strips invalid UTF-8 via iconv
# (which is what loses characters if you decode a token slice on its
# own). Token 4 + 5 spell U+00A2 CENT SIGN across two tokens.
tok_bytes <- list(
  "1" = charToRaw(" hello"),
  "2" = charToRaw(" world"),
  "3" = charToRaw("."),
  "4" = as.raw(0xc2),
  "5" = as.raw(0xa2),
  "6" = charToRaw(" done")
)
stub_decode <- function(ids) {
  b <- unlist(tok_bytes[as.character(ids)], use.names = FALSE)
  if (is.null(b) || length(b) == 0L) {
    return("")
  }
  s <- rawToChar(as.raw(b))
  Encoding(s) <- "UTF-8"
  iconv(s, from = "UTF-8", to = "UTF-8", sub = "")
}
expect_equal(stub_decode(c(4L, 5L)), "¢")
expect_equal(stub_decode(4L), "") # half a character decodes to nothing

la <- function(sd, content, final = FALSE) {
  whisper:::la_advance(sd, as.integer(content), final, stub_decode)
}
ev_types <- function(r) vapply(r$events, function(e)
  if (isTRUE(e$stable)) "stable" else "provisional", character(1))
ev_texts <- function(r) vapply(r$events, function(e) e$text, character(1))

# The design's synthetic sequence: first hypothesis is provisional only;
# agreement commits the shared prefix; final commits the rest
sd <- whisper:::stream_decoder()
r1 <- la(sd, c(1, 2))
expect_equal(ev_types(r1), "provisional")
expect_equal(ev_texts(r1), " hello world")
r2 <- la(r1$sd, c(1, 2, 3))
expect_equal(ev_types(r2), c("stable", "provisional"))
expect_equal(ev_texts(r2), c(" hello world", "."))
r3 <- la(r2$sd, 3, final = TRUE)
expect_equal(ev_types(r3), "stable")
expect_equal(ev_texts(r3), ".")
expect_equal(r3$text, " hello world.")

# An unchanged provisional is not re-sent (REPLACE emits on change only)
sd <- whisper:::stream_decoder()
r1 <- la(sd, 1)
r2 <- la(r1$sd, 1) # same hypothesis: commits " hello", tail empty
expect_equal(ev_types(r2), c("stable", "provisional"))
expect_equal(ev_texts(r2), c(" hello", "")) # provisional cleared
r3 <- la(r2$sd, integer(0)) # still nothing pending: no event at all
expect_equal(length(r3$events), 0L)

# A retracted hypothesis clears the stale provisional with an empty one
sd <- whisper:::stream_decoder()
r1 <- la(sd, 1)
expect_equal(ev_texts(r1), " hello")
r2 <- la(r1$sd, integer(0))
expect_equal(ev_types(r2), "provisional")
expect_equal(ev_texts(r2), "")
r3 <- la(r2$sd, 1) # and it can come back
expect_equal(ev_texts(r3), " hello")

# Unicode boundary: committing between the two bytes of one character
# must not lose it (slice-decoding would emit "" + "" here)
sd <- whisper:::stream_decoder()
r1 <- la(sd, 4)
expect_equal(length(r1$events), 0L) # half a char: nothing to show yet
r2 <- la(r1$sd, c(4, 5)) # agreement commits token 4 alone
expect_equal(ev_types(r2), "provisional")
expect_equal(ev_texts(r2), "¢") # the char is visible as provisional
r3 <- la(r2$sd, 5, final = TRUE) # token 5 commits: char completes
expect_equal(ev_types(r3), "stable")
expect_equal(ev_texts(r3), "¢") # emitted whole, exactly once
expect_equal(r3$sd$stable_sent, "¢")

# Stable concatenation equals the final text exactly across a whole turn
sd <- whisper:::stream_decoder()
stable_all <- character(0)
seqs <- list(c(4), c(4, 5, 1), c(4, 5, 1, 2), c(1, 2, 3))
# note: content passed is beyond committed; simulate via running sd
r <- la(sd, c(4, 5, 1))
stable_all <- c(stable_all, ev_texts(r)[ev_types(r) == "stable"])
r <- la(r$sd, c(4, 5, 1, 2)) # commits 4,5,1 (lcp with prev tail)
stable_all <- c(stable_all, ev_texts(r)[ev_types(r) == "stable"])
r <- la(r$sd, c(2, 3), final = TRUE)
stable_all <- c(stable_all, ev_texts(r)[ev_types(r) == "stable"])
expect_equal(paste(stable_all, collapse = ""), r$text)
expect_equal(r$text, "¢ hello world.")

# --- Stream session mechanics (no model needed) -----------------------

stub_pipe <- structure(list(), class = "whisper_pipeline")

# The declared audio contract is enforced at construction
expect_error(whisper::whisper_stream(stub_pipe, vad = "energy",
  sample_rate = 8000), pattern = "16000")
expect_error(whisper::whisper_stream(stub_pipe, vad = "energy",
  channels = 2), pattern = "mono")
expect_error(whisper::whisper_stream(list()), pattern = "whisper_pipeline")

# end() is terminal: feed() afterwards errors, end() again is a no-op
s <- whisper::whisper_stream(stub_pipe, vad = "energy")
for (n in c(1L, 100L, 511L, 512L, 513L, 4000L)) {
  expect_equal(s$feed(rep(0, n)), list()) # silence never opens a turn
}
expect_equal(s$end(), list())
expect_error(s$feed(rep(0, 512)), pattern = "closed")
expect_equal(s$end(), list())
s$close()

# --- turn speech accumulation and the punctuation gate ----------------

# turn_speech_ms counts speech frames only, from the onset run, and
# resets when the turn ends
sm <- whisper:::vad_sm(min_speech_ms = 250, endpoint_silence_ms = 700)
res <- run_probs(sm, rep(0.9, frames_for(250)))
expect_equal(res$sm$turn_speech_ms, frames_for(250) * frame_ms)
res <- run_probs(res$sm, c(rep(0.9, 10), rep(0.0, 5), rep(0.9, 10)))
expect_equal(res$sm$turn_speech_ms,
  (frames_for(250) + 20) * frame_ms) # the 5 silent frames don't count
res <- run_probs(res$sm, rep(0.0, frames_for(700)))
expect_equal(res$events[[1]]$type, "endpoint")
expect_equal(res$sm$turn_speech_ms, 0)

# silence_target: punctuation only gets authority with enough speech
st <- function(text, speech_ms) {
  whisper:::silence_target(text, speech_ms,
    endpoint_silence_ms = 700, punct_silence_ms = 500,
    midsentence_silence_ms = 1000, punct_min_speech_ms = 1000)
}
expect_equal(st("", 5000), 700) # no transcript: base
expect_equal(st("Ask not what your country can do.", 2000), 500)
expect_equal(st(" Not!", 416), 700) # the JFK fragment: no authority
expect_equal(st("and ask what you", 2000), 1000) # mid-sentence waits
expect_equal(st("and ask what you", 100), 1000) # gate is punct-only
expect_equal(st("Yes.", 999), 700) # just under the gate
expect_equal(st("Yes yes yes yes.", 1000), 500) # at the gate
