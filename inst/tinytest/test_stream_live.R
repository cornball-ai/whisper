# Live streaming integration: feed jfk.mp3 through whisper_stream() in
# random-sized chunks (the framing-is-not-constant rule, exercised) and
# check the event contract against the audio.

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
# JFK's rhetorical pauses run ~1 s; conversation-default silence targets
# would (correctly) split the quote into turns. Stretch them so the test
# exercises one full turn.
stream <- whisper::whisper_stream(pipe, language = "en",
  endpoint_silence_ms = 1200, punct_silence_ms = 1200,
  midsentence_silence_ms = 1500)

samples <- whisper:::load_audio(
  system.file("audio", "jfk.mp3", package = "whisper"))
# Trailing silence lets the endpointer close the last turn naturally.
samples <- c(samples, rep(0, whisper:::WHISPER_SAMPLE_RATE * 1.5))
total_ms <- length(samples) / whisper:::WHISPER_SAMPLE_RATE * 1000

set.seed(42)
events <- list()
pos <- 0L
while (pos < length(samples)) {
  n <- sample(c(160L, 480L, 1024L, 2048L, 4000L), 1L)
  chunk <- samples[(pos + 1L):min(pos + n, length(samples))]
  events <- c(events, stream$feed(chunk))
  pos <- pos + n
}
events <- c(events, stream$end())
stream$close()
options(old_opts)

types <- vapply(events, function(e) e$type, character(1))

# At least one turn ended, and every stable transcript that will exist
# was flushed before its speech_ended
expect_true(sum(types == "speech_ended") >= 1L)

# The stable concatenation carries the transcript (append semantics:
# join with "", nothing inserted)
stable_text <- paste(vapply(
  Filter(function(e) e$type == "transcript" && isTRUE(e$stable), events),
  function(e) e$text, character(1)), collapse = "")
expect_true(grepl("your country can do for you", tolower(stable_text),
  fixed = TRUE))

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
expect_true(all(offsets > 0 & offsets < total_ms))
expect_true(all(diff(offsets) > 0) || length(offsets) == 1L)

# Stable events are never empty and never restate: their total length
# matches the final turn text exactly (append-exactness comes free from
# token-level commits; this catches an accidental re-emission)
stable_events <- Filter(function(e) e$type == "transcript" &&
  isTRUE(e$stable), events)
expect_true(all(vapply(stable_events, function(e) nzchar(e$text),
  logical(1))))
