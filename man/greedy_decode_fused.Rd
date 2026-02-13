% tinyrox says don't edit this manually, but it can't stop you!
\name{greedy_decode_fused}
\alias{greedy_decode_fused}
\title{Greedy Decoding — Fused C++ Path}
\usage{
greedy_decode_fused(
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
Executes the entire decoder forward pass (embedding + all layers +
layer norm + logits + argmax) in a single C++ call per token.
Requires \code{pack_generate_weights()} to have been called on
the decoder.
}
\keyword{internal}
