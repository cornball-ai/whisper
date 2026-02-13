% tinyrox says don't edit this manually, but it can't stop you!
\name{greedy_decode_r}
\alias{greedy_decode_r}
\title{Greedy Decoding — R Path}
\usage{
greedy_decode_r(
  model,
  encoder_output,
  initial_tokens,
  tokenizer,
  max_length = 448L,
  device
)
}
\arguments{
\item{model}{WhisperModel}

\item{encoder_output}{Encoder hidden states}

\item{initial_tokens}{Initial token tensor}

\item{tokenizer}{Tokenizer}

\item{max_length}{Maximum output length}

\item{device}{Device}
}
\description{
Readable reference implementation. Each token goes through the full
R-level decoder forward pass (embedding, per-layer attention + MLP,
layer norm, logits, argmax).
}
\keyword{internal}
