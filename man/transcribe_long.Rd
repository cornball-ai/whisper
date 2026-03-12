% tinyrox says don't edit this manually, but it can't stop you!
\name{transcribe_long}
\alias{transcribe_long}
\title{Transcribe Long Audio}
\usage{
transcribe_long(file, model, tokenizer, config, language, task,
                timestamps = FALSE, word_timestamps = FALSE, beam_size = 1L,
                temperatures = 0, best_of = 1L,
                compression_ratio_threshold = 2.4, logprob_threshold = -1,
                length_penalty = 1, patience = Inf, device, dtype, verbose)
}
\arguments{
\item{file}{Audio file}

\item{model}{WhisperModel}

\item{tokenizer}{Tokenizer}

\item{config}{Model config}

\item{language}{Language}

\item{task}{Task}

\item{timestamps}{Return segment-level timestamps.}

\item{word_timestamps}{Return word-level timestamps.}

\item{beam_size}{Number of beams for beam search.}

\item{temperatures}{Numeric vector of temperatures for fallback.}

\item{best_of}{Number of samples per temperature > 0.}

\item{compression_ratio_threshold}{Max compression ratio before fallback.}

\item{logprob_threshold}{Min average log probability before fallback.}

\item{length_penalty}{Length penalty exponent for beam search.}

\item{patience}{Patience factor for beam search.}

\item{device}{Device}

\item{dtype}{Dtype}

\item{verbose}{Verbose}
}
\value{
Combined transcription result
}
\description{
Process audio longer than 30 seconds in chunks.
}
