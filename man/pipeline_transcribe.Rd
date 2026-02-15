% tinyrox says don't edit this manually, but it can't stop you!
\name{pipeline_transcribe}
\alias{pipeline_transcribe}
\title{Pipeline Transcribe}
\usage{
pipeline_transcribe(
  pipe,
  file,
  language = "en",
  task = "transcribe",
  timestamps = FALSE,
  word_timestamps = FALSE,
  verbose = TRUE
)
}
\arguments{
\item{pipe}{A whisper_pipeline object.}

\item{file}{Path to audio file.}

\item{language}{Language code.}

\item{task}{Task type.}

\item{timestamps}{Return segment-level timestamps.}

\item{word_timestamps}{Return word-level timestamps.}

\item{verbose}{Print progress.}
}
\value{
List with text, language, and metadata.
}
\description{
Pipeline Transcribe
}
\keyword{internal}
