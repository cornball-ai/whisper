% tinyrox says don't edit this manually, but it can't stop you!
\name{.non_speech_token_ids}
\alias{.non_speech_token_ids}
\title{Non-speech token IDs}
\usage{
.non_speech_token_ids(encode)
}
\arguments{
\item{encode}{The tokenizer's \code{encode} function (text -> 0-indexed IDs).}
}
\value{
Sorted integer vector of 0-indexed token IDs.
}
\description{
Mirrors \code{whisper/tokenizer.py::non_speech_tokens}. Encodes a fixed set
of symbols (and their space-prefixed forms); a symbol contributes its first
token when it encodes to a single token, or always for the musical symbols
(whose 3-byte UTF-8 forms share a leading token).
}
\keyword{internal}
