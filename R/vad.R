# vad.R
# Voice activity detection for streaming. Two parts: the Silero VAD model
# (TorchScript, ~2 MB, CPU) that scores 32 ms frames with a speech
# probability, and a pure-R turn state machine that turns those scores into
# onset/endpoint decisions. The state machine never touches torch, so the
# endpointing logic tests without audio or a model.

# Silero v5+ contract at 16 kHz: chunks of exactly 512 samples, internal
# recurrent state, reset between independent audio streams.
VAD_FRAME_SAMPLES <- 512L
VAD_FRAME_MS <- 32 # 512 / 16000 * 1000

# Content-pinned artifact: the URL addresses an immutable commit (tags
# can be moved, commit hashes cannot), and the file's md5 is verified
# before anything is handed to jit_load(), which executes what it loads.
.silero_version <- "v6.2.1"
.silero_commit <- "7e30209a3e901f9842f81b225f3e93d8199902b1"
.silero_md5 <- "cbd961c3faa3246cdd62aefdea2fdbde"

.silero_url <- function() {
  paste0("https://raw.githubusercontent.com/snakers4/silero-vad/",
    .silero_commit, "/src/silero_vad/data/silero_vad.jit")
}

.vad_cache_path <- function() {
  file.path(tools::R_user_dir("whisper", "cache"), "vad",
    paste0("silero_vad_", .silero_version, ".jit"))
}

#' Download the Silero VAD Model
#'
#' Downloads the Silero voice activity detection model (~2 MB, MIT
#' license) used by \code{\link{whisper_stream}} for endpointing. Follows
#' the same consent rules as \code{\link{download_whisper_model}}: asks in
#' interactive sessions, and in non-interactive sessions requires
#' \code{options(whisper.consent = TRUE)}.
#'
#' @param force Re-download even if the model is already cached.
#' @return Path to the cached model file, invisibly.
#' @export
#' @examples
#' \dontrun{
#' download_vad_model()
#' }
download_vad_model <- function(force = FALSE) {
  path <- .vad_cache_path()
  if (!force && file.exists(path)) {
    return(invisible(path))
  }

  # Same consent gate as download_whisper_model (CRAN compliance).
  if (isTRUE(getOption("whisper.consent"))) {
    # Consent already given programmatically
  } else if (interactive()) {
    ans <- utils::askYesNo(
      "Download the Silero VAD model (~2 MB) from GitHub?",
      default = TRUE
    )
    if (!isTRUE(ans)) {
      stop("Download cancelled.", call. = FALSE)
    }
  } else {
    stop(
      "Cannot download the VAD model in non-interactive mode without ",
      "consent. Run download_vad_model() interactively first, or set ",
      "options(whisper.consent = TRUE) to allow downloads.",
      call. = FALSE
    )
  }

  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- tempfile("silero_vad_", tmpdir = dirname(path), fileext = ".tmp")
  on.exit(unlink(tmp), add = TRUE)
  status <- utils::download.file(.silero_url(), tmp, mode = "wb",
    quiet = TRUE)
  if (status != 0L || !file.exists(tmp)) {
    stop("Failed to download the Silero VAD model from ", .silero_url(),
      call. = FALSE)
  }
  got <- unname(tools::md5sum(tmp))
  if (!identical(got, .silero_md5)) {
    stop("Silero VAD download does not match the pinned checksum ",
      "(expected ", .silero_md5, ", got ", got, "). Refusing to load it.",
      call. = FALSE)
  }
  if (!file.rename(tmp, path)) {
    stop("Could not move the downloaded VAD model into place at ", path,
      call. = FALSE)
  }
  message("VAD model downloaded to: ", path)
  invisible(path)
}

# Load the Silero VAD TorchScript module (always CPU: ~1 ms per frame,
# not worth a GPU round trip).
load_vad_model <- function(download = TRUE) {
  path <- .vad_cache_path()
  if (!file.exists(path)) {
    if (!download) {
      stop("Silero VAD model not found. Run download_vad_model() first.",
        call. = FALSE)
    }
    download_vad_model()
  }
  torch::jit_load(path)
}

# Reset Silero's internal recurrent state (between independent streams).
vad_reset <- function(vad) {
  try(vad$reset_states(), silent = TRUE)
  invisible(NULL)
}

# Speech probability for one 512-sample frame of float audio in [-1, 1].
vad_frame_prob <- function(vad, frame) {
  chunk <- torch::torch_tensor(matrix(frame, nrow = 1L),
    dtype = torch::torch_float())
  as.numeric(vad(chunk, torch::jit_scalar(16000L)))
}

# Energy fallback (vad = "energy" in whisper_stream): a graded RMS score
# in [0, 1], honestly weaker than Silero -- a breath and a sentence end
# look identical to it, and it will hold turns open on AEC residue.
energy_frame_prob <- function(frame, full_scale_rms = 0.02) {
  rms <- sqrt(mean(frame ^ 2))
  min(1, rms / full_scale_rms)
}

# --- Turn state machine (pure R) ---------------------------------------
#
# Consumes one speech probability per frame, tracks turn state:
#
#   listening --(speech run >= min_speech_ms)--> in_speech
#   in_speech --(silence run >= silence target)--> endpoint -> listening
#
# Hysteresis: a speech run starts at onset_prob and is extended at
# offset_prob, so a probability hovering between the two never flaps.
# Frames are counted absolutely from the stream start, so every event
# carries sample-exact positions. The silence target is a mutable field
# (`silence_target_ms`) because the stream layer adapts it per tick from
# the transcript tail (punctuation-aware endpointing).

vad_sm <- function(
  onset_prob = 0.5,
  offset_prob = 0.35,
  min_speech_ms = 250,
  endpoint_silence_ms = 700,
  pre_roll_ms = 320
) {
  list(
    onset_prob = onset_prob,
    offset_prob = offset_prob,
    min_speech_ms = min_speech_ms,
    endpoint_silence_ms = endpoint_silence_ms,
    silence_target_ms = endpoint_silence_ms,
    pre_roll_ms = pre_roll_ms,
    state = "listening",
    frame = 0L, # frames consumed so far (absolute)
    run_start = NA_integer_, # first frame of the current speech run
    run_ms = 0, # length of the current speech run (listening)
    silence_ms = 0, # trailing non-speech (in_speech)
    last_speech_frame = NA_integer_, # last frame that scored as speech
    turn_speech_ms = 0 # speech accumulated in the open turn
  )
}

# One frame. Returns list(sm, event) where event is NULL, or
# list(type = "onset", start_frame =) -- start_frame is the first frame of
# the speech run, pre-roll NOT included (the stream layer applies it) --
# or list(type = "endpoint", end_frame =) -- the last speech frame.
vad_sm_step <- function(sm, prob) {
  sm$frame <- sm$frame + 1L
  event <- NULL

  if (sm$state == "listening") {
    in_run <- !is.na(sm$run_start)
    # Written pre-expanded: rformat's expand_if rewrites an if-else nested
    # inside a comparison into branch assignments of the *thresholds*,
    # which are always truthy -- silently turning every frame into speech.
    if (in_run) {
      is_speech <- prob >= sm$offset_prob
    } else {
      is_speech <- prob >= sm$onset_prob
    }
    if (is_speech) {
      if (!in_run) {
        sm$run_start <- sm$frame
        sm$run_ms <- 0
      }
      sm$run_ms <- sm$run_ms + VAD_FRAME_MS
      sm$last_speech_frame <- sm$frame
      if (sm$run_ms >= sm$min_speech_ms) {
        sm$state <- "in_speech"
        sm$silence_ms <- 0
        sm$turn_speech_ms <- sm$run_ms
        event <- list(type = "onset", start_frame = sm$run_start)
      }
    } else {
      sm$run_start <- NA_integer_
      sm$run_ms <- 0
    }
  } else { # in_speech
    if (prob >= sm$offset_prob) {
      sm$silence_ms <- 0
      sm$last_speech_frame <- sm$frame
      sm$turn_speech_ms <- sm$turn_speech_ms + VAD_FRAME_MS
    } else {
      sm$silence_ms <- sm$silence_ms + VAD_FRAME_MS
      if (sm$silence_ms >= sm$silence_target_ms) {
        event <- list(type = "endpoint", end_frame = sm$last_speech_frame)
        sm$state <- "listening"
        sm$run_start <- NA_integer_
        sm$run_ms <- 0
        sm$silence_ms <- 0
        sm$turn_speech_ms <- 0
        sm$silence_target_ms <- sm$endpoint_silence_ms
      }
    }
  }

  list(sm = sm, event = event)
}
