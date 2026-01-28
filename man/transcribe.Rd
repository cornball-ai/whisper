% tinyrox says don't edit this manually, but it can't stop you!
\name{transcribe}
\alias{transcribe}
\title{Whisper Transcription}
\usage{
transcribe(
  file,
  model = "tiny",
  language = "en",
  task = "transcribe",
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

\item{device}{Device: "auto", "cpu", "cuda"}

\item{dtype}{Data type: "auto", "float16", "float32"}

\item{verbose}{Print progress messages}
}
\value{
List with text, language, and metadata
}
\description{
Main transcription API for Whisper.
Transcribe Audio
Transcribe speech from an audio file using Whisper.
}
\examples{
\donttest{
# Transcribe included sample (JFK "ask not" speech)
audio_file <- system.file("audio", "jfk.mp3", package = "whisper")
result <- transcribe(audio_file, model = "tiny")
result$text

# Translate Spanish audio to English
spanish_file <- system.file("audio", "allende.mp3", package = "whisper")
result <- transcribe(spanish_file, model = "tiny",
                     language = "es", task = "translate")
result$text
}
}
