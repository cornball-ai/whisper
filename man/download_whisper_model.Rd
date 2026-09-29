% tinyrox says don't edit this manually, but it can't stop you!
\name{download_whisper_model}
\alias{download_whisper_model}
\title{Download Model from HuggingFace}
\usage{
download_whisper_model(model = "tiny", force = FALSE, revision = NULL)
}
\arguments{
\item{model}{Model name: "tiny", "base", "small", "medium", "large-v3",
"large-v3-turbo"}

\item{force}{Re-download even if exists}

\item{revision}{Optional exact 40-hex commit. Every file is fetched at that
one revision, so the resulting cache entry is a single self-contained
snapshot directory.}
}
\value{
Path to model directory (invisibly)
}
\description{
Download Whisper model weights and tokenizer files from HuggingFace.
In interactive sessions, asks for user consent before downloading.
}
\examples{
\donttest{
if (interactive()) {
  # Download tiny model (smallest, ~150MB)
  download_whisper_model("tiny")

  # Download larger model for better accuracy
  download_whisper_model("small")
}
}
}
