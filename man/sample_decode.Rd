% tinyrox says don't edit this manually, but it can't stop you!
\name{sample_decode}
\alias{sample_decode}
\title{Sample Decode}
\usage{
sample_decode(model, encoder_output, initial_tokens, tokenizer,
              temperature = 0.6, max_length = 448L, timestamps = FALSE,
              word_timestamps = FALSE, device)
}
\arguments{
\item{model}{WhisperModel}

\item{encoder_output}{Encoder hidden states}

\item{initial_tokens}{Initial token tensor (batch=1)}

\item{tokenizer}{Tokenizer}

\item{temperature}{Sampling temperature (must be > 0)}

\item{max_length}{Maximum output length}

\item{timestamps}{Whether to allow timestamp tokens}

\item{word_timestamps}{Whether to collect cross-attention weights}

\item{device}{Device}
}
\value{
List with tokens, cross_attn_weights, sum_logprob, n_tokens
}
\description{
Temperature-scaled sampling decode. Fork of greedy_decode that
uses categorical sampling instead of argmax.
}
