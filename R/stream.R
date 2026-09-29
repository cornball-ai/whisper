# stream.R
# Live streaming session: PCM chunks in, transcript events out. This is
# the in-process capability behind gpu_voice.proto's
# SpeechToText.Transcribe -- the wire, auth, and conversation policy all
# live elsewhere (gpu.ctl and corteza). One session serves one stream and
# many turns; the design doc is tasks/streaming-design.md.

#' Open a Streaming Transcription Session
#'
#' Creates a live transcription session over a loaded
#' \code{\link{whisper_pipeline}}: feed it 16 kHz mono PCM as it arrives
#' and receive transcript events as they are produced, including the
#' endpointing judgment that the speaker's turn is over.
#'
#' The returned object has three methods:
#' \itemize{
#'   \item \code{$feed(pcm)} - consume the next audio chunk. \code{pcm}
#'     is a raw vector (signed 16-bit little-endian PCM), an integer
#'     vector of 16-bit sample values, or a numeric vector of float
#'     samples in [-1, 1]. Chunks can be any length; timing is derived
#'     from the cumulative sample count, never from chunk sizes. Returns
#'     a list of zero or more events.
#'   \item \code{$end()} - end of stream: folds in any sub-frame audio
#'     remainder, flushes any open turn, returns its final events, and
#'     makes the stream terminal (further \code{$feed()} calls error).
#'   \item \code{$close()} - drop buffers and VAD state.
#' }
#'
#' Events are lists with a \code{type} field:
#' \itemize{
#'   \item \code{list(type = "transcript", text =, stable =)} - a stable
#'     transcript APPENDS: the turn's text is the concatenation of stable
#'     events exactly as sent, joining whitespace included. A provisional
#'     transcript (\code{stable = FALSE}) REPLACES the previous
#'     provisional one: it is the whole uncommitted tail.
#'   \item \code{list(type = "speech_ended", audio_offset_ms =)} - the
#'     turn is over. Everything that will ever be stable for the turn is
#'     emitted before this event; the offset points at the end of the
#'     last speech frame in the stream.
#' }
#'
#' Endpointing runs on the Silero VAD model (downloaded on first use, ~2
#' MB, see \code{\link{download_vad_model}}) with an adaptive silence
#' timer: a tail that reads as a finished sentence endpoints after
#' \code{punct_silence_ms} (but only once the turn holds
#' \code{punct_min_speech_ms} of speech -- a shorter turn's transcript is
#' too untrustworthy for its punctuation to shorten the timer), a
#' mid-sentence pause waits \code{midsentence_silence_ms}, and
#' \code{endpoint_silence_ms} applies while there is no transcript yet. Incremental transcription commits
#' what two consecutive decodes agree on (LocalAgreement-2), with
#' committed tokens forced as the decoder prefix so stable text is never
#' contradicted.
#'
#' A turn longer than \code{max_utterance_s} is flushed as a forced
#' endpoint (the turn splits). One live session per process; decoding
#' happens inline in \code{$feed()}.
#'
#' @param pipe A \code{\link{whisper_pipeline}} object.
#' @param language Language code (e.g. "en"), or NULL (default) to detect
#'   once on the first utterance and pin for the session.
#' @param task "transcribe" or "translate".
#' @param sample_rate Declared input sample rate in Hz. Only 16000 is
#'   supported; anything else errors at construction rather than being
#'   silently transcribed at the wrong rate.
#' @param channels Declared input channel count. Only mono (1) is
#'   supported.
#' @param vad "silero" (default) or "energy" (no model download; markedly
#'   weaker endpointing, documented for offline use).
#' @param onset_prob Speech probability that opens a speech run.
#' @param offset_prob Hysteresis threshold that extends one.
#' @param min_speech_ms Shorter blips never open a turn.
#' @param endpoint_silence_ms Base trailing-silence target that ends a
#'   turn.
#' @param punct_silence_ms Silence target when the transcript tail ends a
#'   sentence.
#' @param midsentence_silence_ms Silence target mid-sentence.
#' @param punct_min_speech_ms Speech the turn must hold before terminal
#'   punctuation is allowed to shorten the timer; below it an apparent
#'   sentence end uses \code{endpoint_silence_ms}. A very short turn's
#'   transcript is the least trustworthy evidence in the stream, so its
#'   punctuation gets no authority over the endpoint.
#' @param pre_roll_ms Audio kept from before the detected onset.
#' @param decode_interval_ms New audio per incremental decode.
#' @param max_utterance_s Forced-endpoint cap on a single turn.
#' @param final_temperatures Temperature ladder for the flush decode.
#' @param final_beam_size Beam size for the flush decode.
#' @param download Download the VAD model if missing (consent rules as in
#'   \code{\link{download_vad_model}}).
#' @param verbose Message on turn events.
#' @return A \code{whisper_stream} object.
#' @export
#' @examples
#' \dontrun{
#' pipe <- whisper_pipeline("tiny")
#' stream <- whisper_stream(pipe, language = "en")
#' audio <- whisper:::load_audio(
#'   system.file("audio", "jfk.mp3", package = "whisper"))
#' for (chunk in split(audio, ceiling(seq_along(audio) / 4000))) {
#'   for (ev in stream$feed(chunk)) str(ev)
#' }
#' for (ev in stream$end()) str(ev)
#' }
whisper_stream <- function(
  pipe,
  language = NULL,
  task = "transcribe",
  sample_rate = 16000L,
  channels = 1L,
  vad = c("silero", "energy"),
  onset_prob = 0.5,
  offset_prob = 0.35,
  min_speech_ms = 250,
  endpoint_silence_ms = 700,
  punct_silence_ms = 500,
  midsentence_silence_ms = 1000,
  punct_min_speech_ms = 1000,
  pre_roll_ms = 320,
  decode_interval_ms = 1000,
  max_utterance_s = 28,
  final_temperatures = c(0, 0.2, 0.4, 0.6, 0.8, 1.0),
  final_beam_size = 1L,
  download = TRUE,
  verbose = FALSE
) {
  if (!inherits(pipe, "whisper_pipeline")) {
    stop("`pipe` must be a whisper_pipeline object.", call. = FALSE)
  }
  # The declared audio contract is enforced, not assumed: PCM at the
  # wrong rate transcribes into plausible text with no error, which is
  # exactly why gpu_voice.proto makes config-first mandatory.
  if (!identical(as.integer(sample_rate), WHISPER_SAMPLE_RATE)) {
    stop("Only ", WHISPER_SAMPLE_RATE, " Hz input is supported (got ",
      sample_rate, "); resample before feeding.", call. = FALSE)
  }
  if (!identical(as.integer(channels), 1L)) {
    stop("Only mono input is supported (got ", channels, " channels); ",
      "downmix before feeding.", call. = FALSE)
  }
  vad <- match.arg(vad)

  vad_model <- if (vad == "silero") load_vad_model(download) else NULL
  if (!is.null(vad_model)) vad_reset(vad_model)

  sm <- vad_sm(onset_prob = onset_prob, offset_prob = offset_prob,
    min_speech_ms = min_speech_ms,
    endpoint_silence_ms = endpoint_silence_ms, pre_roll_ms = pre_roll_ms)
  sd <- stream_decoder(language = language, task = task)

  pre_roll_samples <- as.integer(round(pre_roll_ms / 1000 *
    WHISPER_SAMPLE_RATE))
  interval_samples <- as.integer(round(decode_interval_ms / 1000 *
    WHISPER_SAMPLE_RATE))
  max_utt_samples <- as.integer(round(max_utterance_s *
    WHISPER_SAMPLE_RATE))
  # While listening, keep just enough history to serve pre-roll when a
  # run that started keep-ago is confirmed as an onset.
  keep_samples <- pre_roll_samples +
    as.integer(round(min_speech_ms / 1000 * WHISPER_SAMPLE_RATE)) +
    WHISPER_SAMPLE_RATE

  # Session state. `buf` holds recent float audio; buf_start is the
  # absolute (1-based) stream sample index of buf[1]. The stream clock is
  # the cumulative frame count in `sm` (samples = frame * 512).
  st <- new.env(parent = emptyenv())
  st$buf <- numeric(0)
  st$buf_start <- 1
  st$pending <- numeric(0) # sub-frame remainder awaiting a full 512
  st$utt_start <- NA # absolute sample index of the open turn's audio
  st$decoded_upto <- 0L # utterance samples consumed by the last tick
  st$closed <- FALSE

  frame_prob <- function(frame) {
    if (is.null(vad_model)) {
      energy_frame_prob(frame)
    } else {
      vad_frame_prob(vad_model, frame)
    }
  }

  utt_samples <- function() {
    from <- st$utt_start - st$buf_start + 1
    st$buf[from:length(st$buf)]
  }

  run_tick <- function(final = FALSE, samples = utt_samples()) {
    res <- stream_decode_tick(sd, pipe, samples, final = final,
      final_temperatures = final_temperatures,
      final_beam_size = final_beam_size)
    sd <<- res$sd
    # Adapt the silence target to how finished the transcript sounds
    # (punctuation only gets authority once the turn holds enough speech
    # to make its transcript trustworthy).
    sm$silence_target_ms <<- silence_target(res$text, sm$turn_speech_ms,
      endpoint_silence_ms, punct_silence_ms, midsentence_silence_ms,
      punct_min_speech_ms)
    res$events
  }

  flush_turn <- function(end_sample, offset_ms) {
    # Everything that will ever be stable goes out before speech_ended;
    # the flush decode covers audio up to the turn's end sample.
    end_sample <- min(end_sample, st$buf_start + length(st$buf) - 1)
    from <- st$utt_start - st$buf_start + 1
    events <- run_tick(final = TRUE,
      samples = st$buf[from:(end_sample - st$buf_start + 1)])
    events[[length(events) + 1L]] <- list(type = "speech_ended",
      audio_offset_ms = offset_ms)
    if (verbose) message("speech_ended at ", offset_ms, " ms")
    sd <<- stream_decoder_reset_turn(sd)
    st$utt_start <- NA
    st$decoded_upto <- 0L
    sm$silence_target_ms <<- endpoint_silence_ms
    # The finished turn's audio is done with: dropping it both frees the
    # buffer and puts a hard floor under the next turn's pre-roll, so a
    # forced endpoint never decodes the same audio twice.
    drop <- min(end_sample - st$buf_start + 1, length(st$buf))
    if (drop > 0) {
      st$buf <- st$buf[-seq_len(drop)]
      st$buf_start <- st$buf_start + drop
    }
    events
  }

  trim_buf <- function() {
    drop <- length(st$buf) - keep_samples
    if (drop > 0) {
      st$buf <- st$buf[-seq_len(drop)]
      st$buf_start <- st$buf_start + drop
    }
  }

  feed <- function(pcm) {
    if (st$closed) stop("Stream is closed.", call. = FALSE)
    st$pending <- c(st$pending, .pcm_to_float(pcm))
    events <- list()

    n_frames <- length(st$pending) %/% VAD_FRAME_SAMPLES
    for (i in seq_len(n_frames)) {
      frame <- st$pending[((i - 1L) * VAD_FRAME_SAMPLES + 1L):
        (i * VAD_FRAME_SAMPLES)]
      st$buf <- c(st$buf, frame)

      step <- vad_sm_step(sm, frame_prob(frame))
      sm <<- step$sm
      ev <- step$event

      if (!is.null(ev) && ev$type == "onset") {
        onset_sample <- (ev$start_frame - 1) * VAD_FRAME_SAMPLES + 1
        st$utt_start <- max(st$buf_start, onset_sample - pre_roll_samples)
        st$decoded_upto <- 0L
        if (verbose) message("speech onset at ",
          (ev$start_frame - 1) * VAD_FRAME_MS, " ms")
      } else if (!is.null(ev) && ev$type == "endpoint") {
        events <- c(events, flush_turn(ev$end_frame * VAD_FRAME_SAMPLES,
          ev$end_frame * VAD_FRAME_MS))
      }

      if (!is.na(st$utt_start)) {
        utt_len <- st$buf_start + length(st$buf) - st$utt_start
        if (utt_len >= max_utt_samples) {
          # Forced endpoint: a turn the 30 s window cannot hold splits
          # here rather than silently truncating (documented limitation).
          events <- c(events,
            flush_turn(sm$last_speech_frame * VAD_FRAME_SAMPLES,
              sm$last_speech_frame * VAD_FRAME_MS))
          sm$state <<- "listening"
          sm$run_start <<- NA_integer_
          sm$run_ms <<- 0
          sm$silence_ms <<- 0
          sm$turn_speech_ms <<- 0
        } else if (utt_len - st$decoded_upto >= interval_samples) {
          events <- c(events, run_tick())
          st$decoded_upto <- utt_len
        }
      } else {
        trim_buf()
      }
    }
    if (n_frames > 0L) {
      st$pending <- st$pending[-seq_len(n_frames * VAD_FRAME_SAMPLES)]
    }
    events
  }

  end <- function() {
    if (st$closed) {
      return(list())
    }
    events <- if (!is.na(st$utt_start)) {
      # The sub-frame remainder is real audio the VAD never scored; it
      # belongs to the flush decode, at its true sample count.
      if (length(st$pending) > 0L) {
        st$buf <- c(st$buf, st$pending)
        st$pending <- numeric(0)
      }
      end_sample <- st$buf_start + length(st$buf) - 1
      flush_turn(end_sample,
        round(end_sample / WHISPER_SAMPLE_RATE * 1000))
    } else {
      list()
    }
    st$closed <- TRUE # terminal: the stream is over, feed() now errors
    events
  }

  close <- function() {
    st$closed <- TRUE
    st$buf <- numeric(0)
    st$pending <- numeric(0)
    if (!is.null(vad_model)) vad_reset(vad_model)
    invisible(NULL)
  }

  obj <- list(feed = feed, end = end, close = close)
  class(obj) <- "whisper_stream"
  obj
}

#' @export
print.whisper_stream <- function(x, ...) {
  cat("<whisper_stream>\n")
  invisible(x)
}

# S16LE bytes or int16 values -> float [-1, 1]; float passes through.
.pcm_to_float <- function(pcm) {
  if (is.raw(pcm)) {
    if (length(pcm) %% 2L != 0L) {
      stop("Raw PCM must be a whole number of 16-bit samples.",
        call. = FALSE)
    }
    readBin(pcm, "integer", n = length(pcm) %/% 2L, size = 2L,
      signed = TRUE, endian = "little") / 32768
  } else if (is.integer(pcm)) {
    pcm / 32768
  } else if (is.numeric(pcm)) {
    pcm
  } else {
    stop("`pcm` must be raw, integer, or numeric.", call. = FALSE)
  }
}
