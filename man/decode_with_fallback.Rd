% tinyrox says don't edit this manually, but it can't stop you!
\name{decode_with_fallback}
\alias{decode_with_fallback}
\title{Decode with Temperature Fallback}
\usage{
decode_with_fallback(model, encoder_output, initial_tokens, tokenizer,
                     temperatures = c(0, 0.2, 0.4, 0.6, 0.8, 1),
                     beam_size = 5L, best_of = 5L, max_length = 448L,
                     timestamps = FALSE, word_timestamps = FALSE,
                     compression_ratio_threshold = 2.4, logprob_threshold = -1,
                     length_penalty = 1, patience = Inf, device)
}
\arguments{
\item{model}{WhisperModel}

\item{encoder_output}{Encoder hidden states}

\item{initial_tokens}{Initial token tensor}

\item{tokenizer}{Tokenizer}

\item{temperatures}{Numeric vector of temperatures to try}

\item{beam_size}{Number of beams for temp=0}

\item{best_of}{Number of samples for temp>0}

\item{max_length}{Maximum output length}

\item{timestamps}{Whether to allow timestamp tokens}

\item{word_timestamps}{Whether to collect cross-attention weights}

\item{compression_ratio_threshold}{Max compression ratio}

\item{logprob_threshold}{Min average log probability}

\item{length_penalty}{Length penalty for beam search}

\item{patience}{Patience factor for beam search}

\item{device}{Device}
}
\value{
List with tokens, cross_attn_weights, sum_logprob, n_tokens
}
\description{
Try decoding at progressively higher temperatures until quality
thresholds are met. At temperature 0, uses beam search (or greedy
if beam_size=1). At temperature > 0, uses sampling with best-of.
}
