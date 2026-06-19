% tinyrox says don't edit this manually, but it can't stop you!
\name{.decode_suppress_ids}
\alias{.decode_suppress_ids}
\title{Token IDs suppressed at every decode step}
\usage{
.decode_suppress_ids(encode, special)
}
\arguments{
\item{encode}{The tokenizer's \code{encode} function.}

\item{special}{Named list of special token IDs.}
}
\value{
Sorted integer vector of 0-indexed token IDs.
}
\description{
\code{non_speech_tokens} plus the control tokens the reference also adds in
\code{decoding.py::_get_suppress_tokens}: translate, transcribe, sot_lm,
sot_prev, sot, and no_speech.
}
\keyword{internal}
