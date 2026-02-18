% tinyrox says don't edit this manually, but it can't stop you!
\name{transcribe}
\alias{transcribe}
\title{Transcribe Audio}
\usage{
transcribe(
  file,
  model = "tiny",
  language = "en",
  task = "transcribe",
  timestamps = FALSE,
  word_timestamps = FALSE,
  device = "auto",
  dtype = "auto",
  verbose = TRUE
)
}
\arguments{
\item{file}{Path to audio file (WAV, MP3, etc.)}

\item{model}{Model name: "tiny", "base", "small", "medium", "large-v3"}

\item{language}{Language code (e.g., "en", "es"). NULL for auto-detection.}

\item{task}{"transcribe" or "translate" (translate to English)}

\item{timestamps}{If TRUE, return segment-level timestamps}

\item{word_timestamps}{If TRUE, return word-level timestamps (implies timestamps)}

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
# Transcribe included sample (JFK "ask not" speech)
if (model_exists("tiny")) {
  audio_file <- system.file("audio", "jfk.mp3", package = "whisper")
  result <- transcribe(audio_file, model = "tiny")
  result$text

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
