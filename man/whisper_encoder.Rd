% tinyrox says don't edit this manually, but it can't stop you!
\name{whisper_encoder}
\alias{whisper_encoder}
\title{Audio Encoder}
\usage{
whisper_encoder(n_mels, n_ctx, n_state, n_head, n_layer)
}
\arguments{
\item{n_mels}{Number of mel spectrogram bins}

\item{n_ctx}{Maximum context length (1500 for 30s audio)}

\item{n_state}{Hidden dimension}

\item{n_head}{Number of attention heads}

\item{n_layer}{Number of transformer layers}
}
\description{
Full Whisper encoder: Conv stem + positional encoding + transformer layers.
}
