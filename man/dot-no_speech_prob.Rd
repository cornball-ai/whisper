% tinyrox says don't edit this manually, but it can't stop you!
\name{.no_speech_prob}
\alias{.no_speech_prob}
\title{No-speech probability from the prompt prefill}
\usage{
.no_speech_prob(logits, generated, special)
}
\arguments{
\item{logits}{Prefill logits, shape \code{(1, n_prompt, vocab)}.}

\item{generated}{Integer vector of the prompt tokens (to locate SOT).}

\item{special}{Named list of special token IDs.}
}
\value{
Scalar no-speech probability (numeric).
}
\description{
Mirrors \code{decoding.py}: the probability mass on the \code{<|nospeech|>}
token in the softmax taken at the SOT position of the prefill logits (the
distribution that chooses the language slot). High values mean the window is
likely silence.
}
\keyword{internal}
