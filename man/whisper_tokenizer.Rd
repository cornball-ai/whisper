% tinyrox says don't edit this manually, but it can't stop you!
\name{whisper_tokenizer}
\alias{whisper_tokenizer}
\title{Whisper BPE Tokenizer}
\usage{
whisper_tokenizer(model = "tiny")
}
\arguments{
\item{model}{Model name for vocab lookup}
}
\value{
Tokenizer object (list with encode/decode functions)
}
\description{
Byte-pair encoding tokenizer for Whisper models.
Create Whisper Tokenizer
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
