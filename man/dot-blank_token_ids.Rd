% tinyrox says don't edit this manually, but it can't stop you!
\name{.blank_token_ids}
\alias{.blank_token_ids}
\title{Token IDs suppressed at the first decode step (SuppressBlank)}
\usage{
.blank_token_ids(encode, special)
}
\arguments{
\item{encode}{The tokenizer's \code{encode} function.}

\item{special}{Named list of special token IDs.}
}
\value{
Sorted integer vector of 0-indexed token IDs.
}
\description{
Token IDs suppressed at the first decode step (SuppressBlank)
}
\keyword{internal}
