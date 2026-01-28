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
  verbose = TRUE
)
}
\arguments{
\item{model}{Model name: "tiny", "base", "small", "medium", "large-v3"}

\item{device}{Device to load model on ("auto", "cpu", "cuda")}

\item{dtype}{Data type ("auto", "float16", "float32")}

\item{download}{If TRUE and model not present, prompt to download}

\item{verbose}{Print loading messages}
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
