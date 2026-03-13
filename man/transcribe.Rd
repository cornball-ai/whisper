% tinyrox says don't edit this manually, but it can't stop you!
\name{transcribe}
\alias{transcribe}
\title{Transcribe Audio}
\usage{
transcribe(file, model = "tiny", language = NULL, task = "transcribe",
           timestamps = FALSE, word_timestamps = FALSE, beam_size = 1L,
           temperatures = 0, best_of = 1L, compression_ratio_threshold = 2.4,
           logprob_threshold = -1, length_penalty = 1, patience = Inf,
           device = "auto", dtype = "auto", verbose = TRUE)
}
\arguments{
\item{file}{Path to audio file (WAV, MP3, etc.)}

\item{model}{Model name: "tiny", "base", "small", "medium", "large-v3"}

\item{language}{Language code (e.g., "en", "es"), or NULL (default) for
auto-detection from the audio.}

\item{task}{"transcribe" or "translate" (translate to English)}

\item{timestamps}{If TRUE, return segment-level timestamps}

\item{word_timestamps}{If TRUE, return word-level timestamps (implies timestamps)}

\item{beam_size}{Number of beams for beam search (1 = greedy, default)}

\item{temperatures}{Numeric vector of temperatures to try. 0 uses beam search
or greedy; values > 0 use sampling. Multiple values enable fallback.}

\item{best_of}{Number of samples per temperature > 0, keeping the best.}

\item{compression_ratio_threshold}{Max compression ratio before fallback.}

\item{logprob_threshold}{Min average log probability before fallback.}

\item{length_penalty}{Length penalty exponent for beam search scoring.}

\item{patience}{Patience factor for beam search (stop after patience*beam_size).}

\item{device}{Device: "auto", "cpu", "cuda"}

\item{dtype}{Data type: "auto", "float16", "float32"}

\item{verbose}{Print progress messages}
}
\value{
List with text, language, and metadata. When \code{timestamps=TRUE},
  includes \code{segments} data.frame with start, end, text columns. When
  \code{word_timestamps=TRUE}, includes \code{words} data.frame with word,
  start, end columns.
}
\description{
Transcribe speech from an audio file using Whisper.
For repeated transcription, use \code{\link{whisper_pipeline}()} to
load the model once.
}
\examples{
\donttest{
if (model_exists("tiny")) {
  audio_file <- system.file("audio", "jfk.mp3", package = "whisper")

  # Auto-detect language (default)
  result <- transcribe(audio_file, model = "tiny")
  result$language  # "en"
  result$text

  # Explicit language
  result <- transcribe(audio_file, model = "tiny", language = "en")

  # With timestamps
  result <- transcribe(audio_file, model = "tiny", timestamps = TRUE)
  result$segments

  # Translate Spanish audio to English
  spanish_file <- system.file("audio", "allende.mp3", package = "whisper")
  result <- transcribe(spanish_file, model = "tiny",
                       language = "es", task = "translate")
  result$text
}
}
}
