% tinyrox says don't edit this manually, but it can't stop you!
\name{compute_stft}
\alias{compute_stft}
\title{Compute STFT Magnitude}
\usage{
compute_stft(audio, n_fft = WHISPER_N_FFT, hop_length = WHISPER_HOP_LENGTH)
}
\arguments{
\item{audio}{Numeric vector of audio samples}

\item{n_fft}{FFT window size}

\item{hop_length}{Hop length between frames}
}
\value{
Complex STFT matrix
}
\description{
Compute STFT Magnitude
}
