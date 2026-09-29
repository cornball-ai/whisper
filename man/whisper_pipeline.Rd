% tinyrox says don't edit this manually, but it can't stop you!
\name{whisper_pipeline}
\alias{whisper_pipeline}
\title{Create a Whisper Pipeline}
\usage{
whisper_pipeline(
  model = "tiny",
  device = "auto",
  dtype = "auto",
  download = TRUE,
  verbose = TRUE,
  revision = NULL
)
}
\arguments{
\item{model}{Model name: "tiny", "base", "small", "medium", "large-v3",
"large-v3-turbo"}

\item{device}{Device: "auto", "cpu", "cuda"}

\item{dtype}{Data type: "auto", "float16", "float32"}

\item{download}{If TRUE and model not present, prompt to download.}

\item{verbose}{Print loading messages.}

\item{revision}{Optional exact 40-hex commit. Weights, config and tokenizer
all resolve at that one revision, so the pipeline builds against a cache
holding only that snapshot directory.}
}
\value{
A \code{whisper_pipeline} object with a \code{$transcribe()} method.
}
\description{
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
