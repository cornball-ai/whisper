% tinyrox says don't edit this manually, but it can't stop you!
\name{pipeline_transcribe}
\alias{pipeline_transcribe}
\title{Pipeline Transcribe}
\usage{
pipeline_transcribe(pipe, file, language = NULL, task = "transcribe",
                    timestamps = FALSE, word_timestamps = FALSE,
                    beam_size = 1L, temperatures = 0, best_of = 1L,
                    compression_ratio_threshold = 2.4, logprob_threshold = -1,
                    length_penalty = 1, patience = Inf, verbose = TRUE)
}
\arguments{
\item{pipe}{A whisper_pipeline object.}

\item{file}{Path to audio file.}

\item{language}{Language code.}

\item{task}{Task type.}

\item{timestamps}{Return segment-level timestamps.}

\item{word_timestamps}{Return word-level timestamps.}

\item{beam_size}{Number of beams for beam search.}

\item{temperatures}{Numeric vector of temperatures for fallback.}

\item{best_of}{Number of samples per temperature > 0.}

\item{compression_ratio_threshold}{Max compression ratio before fallback.}

\item{logprob_threshold}{Min average log probability before fallback.}

\item{length_penalty}{Length penalty exponent for beam search.}

\item{patience}{Patience factor for beam search.}

\item{verbose}{Print progress.}
}
\value{
List with text, language, and metadata.
}
\description{
Pipeline Transcribe
}
\keyword{internal}
