% tinyrox says don't edit this manually, but it can't stop you!
\name{load_whisper_model}
\alias{load_whisper_model}
\title{Load Whisper Model}
\usage{
load_whisper_model(
  model = "tiny",
  device = "auto",
  dtype = "auto",
  download = FALSE,
  verbose = TRUE,
  revision = NULL
)
}
\arguments{
\item{model}{Model name: "tiny", "base", "small", "medium", "large-v3",
"large-v3-turbo"}

\item{device}{Device to load model on ("auto", "cpu", "cuda")}

\item{dtype}{Data type ("auto", "float16", "float32")}

\item{download}{If TRUE and model not present, prompt to download}

\item{verbose}{Print loading messages}

\item{revision}{Optional exact 40-hex commit to load. With one, the weights
resolve straight out of \code{snapshots/<revision>/}, so loading works
against a cache that holds only that snapshot -- no \code{refs/} entry and
no network.}
}
\value{
WhisperModel module
}
\description{
Load a Whisper model with weights from HuggingFace.
}
\examples{
\donttest{
# Load tiny model (requires prior download)
if (model_exists("tiny")) {
  model <- load_whisper_model("tiny")
}
}
}
