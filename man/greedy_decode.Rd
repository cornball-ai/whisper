% tinyrox says don't edit this manually, but it can't stop you!
\name{greedy_decode}
\alias{greedy_decode}
\title{Greedy Decoding}
\usage{
greedy_decode(
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
\value{
Integer vector of generated tokens
}
\description{
Greedy Decoding
}
