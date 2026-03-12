% tinyrox says don't edit this manually, but it can't stop you!
\name{transcribe_chunk}
\alias{transcribe_chunk}
\title{Transcribe Single Chunk}
\usage{
transcribe_chunk(file, model, tokenizer, config, language = NULL,
                 task = "transcribe", timestamps = FALSE,
                 word_timestamps = FALSE, beam_size = 1L, temperatures = 0,
                 best_of = 1L, compression_ratio_threshold = 2.4,
                 logprob_threshold = -1, length_penalty = 1, patience = Inf,
                 time_offset = 0, device, dtype, verbose = TRUE)
}
\arguments{
\item{file}{Audio file or mel spectrogram}

\item{model}{WhisperModel}

\item{tokenizer}{Tokenizer}

\item{config}{Model config}

\item{language}{Language code}

\item{task}{Task type}

\item{timestamps}{Return segment-level timestamps.}

\item{word_timestamps}{Return word-level timestamps.}

\item{beam_size}{Number of beams for beam search.}

\item{temperatures}{Numeric vector of temperatures for fallback.}

\item{best_of}{Number of samples per temperature > 0.}

\item{compression_ratio_threshold}{Max compression ratio before fallback.}

\item{logprob_threshold}{Min average log probability before fallback.}

\item{length_penalty}{Length penalty exponent for beam search.}

\item{patience}{Patience factor for beam search.}

\item{time_offset}{Time offset in seconds for chunk processing.}

\item{device}{Device}

\item{dtype}{Dtype}

\item{verbose}{Verbose output}
}
\value{
Transcription result
}
\description{
Transcribe Single Chunk
}
