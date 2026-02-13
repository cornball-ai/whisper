% tinyrox says don't edit this manually, but it can't stop you!
\name{whisper_pipeline}
\alias{whisper_pipeline}
\title{Whisper Transcription}
\usage{
whisper_pipeline(
  model = "tiny",
  device = "auto",
  dtype = "auto",
  download = TRUE,
  verbose = TRUE
)
}
\arguments{
\item{model}{Model name: "tiny", "base", "small", "medium", "large-v3"}

\item{device}{Device: "auto", "cpu", "cuda"}

\item{dtype}{Data type: "auto", "float16", "float32"}

\item{download}{If TRUE and model not present, prompt to download.}

\item{verbose}{Print loading messages.}
}
\value{
A \code{whisper_pipeline} object with a \code{$transcribe()} method.
}
\description{
Main transcription API for Whisper.
Create a Whisper Pipeline
Load the model, tokenizer, and config once. Call \code{$transcribe()}
repeatedly without reloading.
}
\examples{
\donttest{
if (model_exists("tiny")) {
  pipe <- whisper_pipeline("tiny")
  pipe$transcribe(system.file("audio", "jfk.mp3", package = "whisper"))
}
}
}
