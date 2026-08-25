# Live streaming integration: real audio through whisper_stream() in
# random-sized chunks (the framing-is-not-constant rule, exercised),
# checked against the event contract, batch output, chunk-boundary
# invariance, multiple turns, and non-ASCII text.

if (!requireNamespace("torch", quietly = TRUE) ||
    !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!at_home()) {
  exit_file("live streaming test runs at home only")
}
if (!whisper::model_exists("tiny")) {
  exit_file("tiny model not downloaded")
}

old_opts <- options(whisper.consent = TRUE) # allow the VAD download
pipe <- whisper::whisper_pipeline("tiny", verbose = FALSE)

feed_chunked <- function(stream, samples, seed) {
  set.seed(seed)
  events <- list()
  pos <- 0L
  while (pos < length(samples)) {
    n <- sample(c(160L, 480L, 1024L, 2048L, 4000L), 1L)
    chunk <- samples[(pos + 1L):min(pos + n, length(samples))]
    events <- c(events, stream$feed(chunk))
    pos <- pos + n
  }
  events
}
stable_concat <- function(events) {
  paste(vapply(
    Filter(function(e) e$type == "transcript" && isTRUE(e$stable), events),
    function(e) e$text, character(1)), collapse = "")
}
check_contract <- function(events, total_ms) {
  types <- vapply(events, function(e) e$type, character(1))
  # Flush-before-SpeechEnded: the transcript event nearest before each
  # speech_ended is stable, never provisional
  for (i in which(types == "speech_ended")) {
    prior <- rev(seq_len(i - 1L))
    prior_transcripts <- prior[types[prior] == "transcript"]
    if (length(prior_transcripts) > 0L) {
      expect_true(isTRUE(events[[prior_transcripts[1L]]]$stable))
    }
  }
  # Offsets are sane and ascending
  offsets <- vapply(Filter(function(e) e$type == "speech_ended", events),
    function(e) e$audio_offset_ms, numeric(1))
  expect_true(all(offsets > 0 & offsets <= total_ms))
  expect_true(all(diff(offsets) > 0) || length(offsets) <= 1L)
  # Stable events never carry empty text (an empty event may only be a
  # provisional retraction)
  expect_true(all(vapply(
    Filter(function(e) e$type == "transcript" && isTRUE(e$stable), events),
    function(e) nzchar(e$text), logical(1))))
  invisible(offsets)
}

jfk <- whisper:::load_audio(
  system.file("audio", "jfk.mp3", package = "whisper"))
sr <- whisper:::WHISPER_SAMPLE_RATE

# --- one full turn: contract + batch comparison -----------------------
# JFK's rhetorical pauses run ~1 s; conversation-default silence targets
# would (correctly) split the quote into turns. Stretch them so this run
# exercises one full turn.
one_turn <- c(jfk, rep(0, sr * 1.5))
stream <- whisper::whisper_stream(pipe, language = "en",
  endpoint_silence_ms = 1200, punct_silence_ms = 1200,
  midsentence_silence_ms = 1500)
events_a <- feed_chunked(stream, one_turn, seed = 42)
events_a <- c(events_a, stream$end())
stream$close()

types_a <- vapply(events_a, function(e) e$type, character(1))
expect_true(sum(types_a == "speech_ended") >= 1L)
check_contract(events_a, length(one_turn) / sr * 1000)

stable_a <- stable_concat(events_a)
expect_true(grepl("your country can do for you", tolower(stable_a),
  fixed = TRUE))

# The streamed transcript carries substantially the batch transcript
# (the flush decode is prefix-conditioned, so exact equality is not
# guaranteed; length parity plus the phrase check above is)
batch_text <- pipe$transcribe(
  system.file("audio", "jfk.mp3", package = "whisper"),
  language = "en", verbose = FALSE)$text
expect_true(nchar(stable_a) > 0.6 * nchar(batch_text))

# --- chunk-boundary invariance ----------------------------------------
# Different chunking of the same PCM is the same sample stream, so the
# events must be identical text
stream <- whisper::whisper_stream(pipe, language = "en",
  endpoint_silence_ms = 1200, punct_silence_ms = 1200,
  midsentence_silence_ms = 1500)
events_b <- feed_chunked(stream, one_turn, seed = 7)
events_b <- c(events_b, stream$end())
stream$close()
expect_equal(stable_concat(events_b), stable_a)

# --- two turns, second flushed by $end() with a sub-frame remainder ----
# jfk.mp3 is 120560 samples = 235 frames + a 240-sample remainder, so
# ending mid-turn exercises the remainder fold-in
two_turns <- c(jfk, rep(0, sr * 1.5), jfk)
stream <- whisper::whisper_stream(pipe, language = "en",
  endpoint_silence_ms = 1200, punct_silence_ms = 1200,
  midsentence_silence_ms = 1500)
events_c <- feed_chunked(stream, two_turns, seed = 42)
events_c <- c(events_c, stream$end())

types_c <- vapply(events_c, function(e) e$type, character(1))
expect_true(sum(types_c == "speech_ended") >= 2L)
check_contract(events_c, length(two_turns) / sr * 1000)
# Both turns transcribed; the turn boundary resets cleanly
expect_true(nchar(stable_concat(events_c)) > 1.5 * nchar(stable_a))

# $end() is terminal
expect_error(stream$feed(rep(0, 512)), pattern = "closed")
expect_equal(stream$end(), list())
stream$close()

# --- non-ASCII text survives the stable/append path -------------------
# Spanish audio: accented characters are multibyte UTF-8, the case where
# slice-decoding at commit boundaries loses characters. Whether tiny
# hears any accented word in this clip is the model's business, so the
# preservation check calibrates against an unstreamed decode of the same
# slice rather than hard-coding vocabulary luck.
allende <- whisper:::load_audio(
  system.file("audio", "allende.mp3", package = "whisper"))
slice <- allende[seq_len(24L * sr)]
ref_text <- whisper:::stream_decode_tick(
  whisper:::stream_decoder(language = "es"), pipe, slice,
  final = TRUE)$text

spanish <- c(slice, rep(0, sr * 1.5))
stream <- whisper::whisper_stream(pipe, language = "es",
  endpoint_silence_ms = 1200, punct_silence_ms = 1200,
  midsentence_silence_ms = 1500)
events_d <- feed_chunked(stream, spanish, seed = 42)
events_d <- c(events_d, stream$end())
stream$close()
options(old_opts)

stable_d <- stable_concat(events_d)
expect_true(nchar(stable_d) > 0)
expect_false(grepl("�", stable_d)) # no replacement characters
if (any(utf8ToInt(ref_text) > 127L)) {
  expect_true(any(utf8ToInt(stable_d) > 127L)) # accents made it through
}
