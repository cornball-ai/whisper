% tinyrox says don't edit this manually, but it can't stop you!
\name{.suppress_mask}
\alias{.suppress_mask}
\title{Build an additive logit mask (-Inf at suppressed positions, 0 elsewhere)}
\usage{
.suppress_mask(ids0, n_vocab, device, dtype)
}
\arguments{
\item{ids0}{0-indexed token IDs to suppress.}

\item{n_vocab}{Vocabulary size (logit width).}

\item{device}{torch device.}

\item{dtype}{torch dtype.}
}
\value{
A \code{(1, n_vocab)} tensor.
}
\description{
Added to a \code{(1, n_vocab)} logit row before argmax. Using an additive
mask avoids tensor advanced-index assignment and works on any device/dtype.
}
\keyword{internal}
