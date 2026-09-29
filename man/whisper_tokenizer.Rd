% tinyrox says don't edit this manually, but it can't stop you!
\name{whisper_tokenizer}
\alias{whisper_tokenizer}
\title{Create Whisper Tokenizer}
\usage{
whisper_tokenizer(model = "tiny", revision = NULL)
}
\arguments{
\item{model}{Model name for vocab lookup}

\item{revision}{Optional exact 40-hex commit. The vocab files resolve out of
that snapshot directory, matching weights loaded at the same revision.}
}
\value{
Tokenizer object (list with encode/decode functions)
}
\description{
Load or create a Whisper tokenizer from HuggingFace vocab files.
}
\examples{
\donttest{
# Load tokenizer (requires prior model download)
if (model_exists("tiny")) {
  tok <- whisper_tokenizer("tiny")
  tok$encode("Hello world")
  tok$decode(c(50258, 50259, 50359, 50363))
}
}
}
