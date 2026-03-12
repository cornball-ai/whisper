% tinyrox says don't edit this manually, but it can't stop you!
\name{beam_search_decode}
\alias{beam_search_decode}
\title{Beam Search Decode}
\usage{
beam_search_decode(model, encoder_output, initial_tokens, tokenizer,
                   beam_size = 5L, max_length = 448L, timestamps = FALSE,
                   word_timestamps = FALSE, length_penalty = 1, patience = Inf,
                   device)
}
\arguments{
\item{model}{WhisperModel}

\item{encoder_output}{Encoder hidden states (batch=1)}

\item{initial_tokens}{Initial token tensor (batch=1)}

\item{tokenizer}{Tokenizer}

\item{beam_size}{Number of beams}

\item{max_length}{Maximum output length}

\item{timestamps}{Whether to allow timestamp tokens}

\item{word_timestamps}{Whether to collect cross-attention weights}

\item{length_penalty}{Length penalty exponent}

\item{patience}{Patience factor (stop after patience*beam_size finished)}

\item{device}{Device}
}
\value{
List with tokens, cross_attn_weights, sum_logprob, n_tokens
}
\description{
Beam search decoding for Whisper. Maintains multiple hypotheses
and selects the best one based on length-normalized log probability.
}
