% tinyrox says don't edit this manually, but it can't stop you!
\name{detect_language_from_mel}
\alias{detect_language_from_mel}
\title{Detect Language from Mel Spectrogram}
\usage{
detect_language_from_mel(model, mel, config, device, top_k = 5L)
}
\arguments{
\item{model}{WhisperModel}

\item{mel}{Mel spectrogram tensor}

\item{config}{Model config}

\item{device}{torch device}

\item{top_k}{Number of top probabilities to return}
}
\value{
List with language code and probabilities
}
\description{
Core detection logic. Feed SOT token to decoder, read language logits.
}
