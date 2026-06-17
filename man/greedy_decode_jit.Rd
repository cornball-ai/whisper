% tinyrox says don't edit this manually, but it can't stop you!
\name{greedy_decode_jit}
\alias{greedy_decode_jit}
\title{Greedy Decoding with a TorchScript decode loop}
\usage{
greedy_decode_jit(model, encoder_output, initial_tokens, tokenizer,
                  max_length = 448L, timestamps = FALSE,
                  word_timestamps = FALSE, device)
}
\arguments{
\item{model}{WhisperModel}

\item{encoder_output}{Encoder hidden states}

\item{initial_tokens}{Initial token tensor (batch=1)}

\item{tokenizer}{Tokenizer}

\item{max_length}{Maximum output length}

\item{timestamps}{Whether to allow timestamp tokens}

\item{word_timestamps}{Whether to collect cross-attention weights}

\item{device}{Device}
}
\value{
List with tokens, cross_attn_weights, sum_logprob, n_tokens
}
\description{
Token-for-token equivalent of \code{\link{greedy_decode}}: eager
prefill on the initial prompt, then each new token's decoder forward
runs as one \code{jit_compile}'d TorchScript call. The self-attention
KV cache is pre-allocated to \code{max_length} and the cross-attention
K/V are cached once from the encoder output. When \code{word_timestamps}
is TRUE it uses the cross-attention-weight variant of the step (manual
softmax cross-attention) and collects the per-token weights in the same
order as the eager path, so word-level DTW alignment works on the JIT
path too.
}
\keyword{internal}
