#' Format Seconds as an SRT/ASS Timestamp
#'
#' Renders numeric seconds as `"HH:MM:SS.mmm"`, the timestamp form
#' subtitle tooling expects. Rounds to whole milliseconds first, so a value
#' just under a minute boundary carries into the minutes field instead of
#' printing a 60th second.
#'
#' Negative and non-finite input (`NA`, `NaN`, `Inf`, `-Inf`) clamps to zero.
#' `sprintf()`'s integer conversions reject non-finite doubles outright, so
#' the guard has to catch infinities and not just missings.
#'
#' @param t Numeric vector of seconds.
#'
#' @return Character vector of `"HH:MM:SS.mmm"` timestamps.
#'
#' @noRd
format_hms <- function(t) {
  ms <- round(as.numeric(t) * 1000)
  ms[!is.finite(ms) | ms < 0] <- 0
  sprintf("%02d:%02d:%06.3f", ms %/% 3600000, (ms %% 3600000) %/% 60000,
    (ms %% 60000) / 1000)
}

#' Attach the Subtitle-Tool Shape to a Transcription Result
#'
#' The `subtitles` package's `whisper_to_srt()` and
#' `whisper_to_ass()` take an `audio.whisper`-shaped object: class
#' `"whisper_transcription"`, carrying a `data` frame of `from`/`to`
#' timestamp strings and `text`. This attaches that shape to a
#' `transcribe()` result so it feeds those functions directly.
#'
#' Purely additive. `text`, `segments`, `words`, and the rest are left
#' untouched, so callers reading `x$segments` are unaffected. Results without
#' segments (`timestamps = FALSE`) are returned unchanged and unclassed --
#' there is nothing to build a subtitle file from, and claiming the class
#' without `data` would fail inside the subtitle writers.
#'
#' The class vector leads with `"whisper_result"` so this package's own S3
#' methods get first dispatch. `audio.whisper` also defines methods on
#' `"whisper_transcription"`; leading with our own class keeps those as a
#' fallback rather than letting them capture our objects when both packages
#' are attached.
#'
#' @param result A `transcribe()` result list.
#'
#' @return `result`, with `data` added and the class set, when it has
#'   segments; otherwise `result` unchanged.
#'
#' @noRd
attach_subtitle_shape <- function(result) {
  segs <- result$segments
  if (is.null(segs) || nrow(segs) == 0) {
    return(result)
  }

  result$data <- data.frame(
    from = format_hms(segs$start),
    to = format_hms(segs$end),
    text = segs$text,
    stringsAsFactors = FALSE)
  class(result) <- c("whisper_result", "whisper_transcription")
  result
}

#' Print a Transcription Result
#'
#' @param x A `transcribe()` result.
#' @param ... Ignored.
#'
#' @return `x`, invisibly.
#'
#' @export
print.whisper_result <- function(x, ...) {
  cat("<whisper transcription>\n")
  cat("  model:    ", x$model, "\n", sep = "")
  cat("  language: ", x$language, "\n", sep = "")
  if (!is.null(x$duration)) {
    cat("  duration: ", sprintf("%.2fs", x$duration), "\n", sep = "")
  }
  cat("  segments: ", nrow(x$segments), "\n", sep = "")
  if (!is.null(x$words)) {
    cat("  words:    ", nrow(x$words), "\n", sep = "")
  }
  cat("\n", strwrap(x$text, width = 0.9 * getOption("width")), sep = "\n")
  invisible(x)
}
