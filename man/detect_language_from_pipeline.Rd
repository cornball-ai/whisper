% tinyrox says don't edit this manually, but it can't stop you!
\name{detect_language_from_pipeline}
\alias{detect_language_from_pipeline}
\title{Detect Language from Pipeline}
\usage{
detect_language_from_pipeline(pipe, file, top_k = 5L)
}
\arguments{
\item{pipe}{A whisper_pipeline object}

\item{file}{Path to audio file, or numeric vector of audio samples}

\item{top_k}{Number of top probabilities to return}
}
\value{
List with language code and probabilities
}
\description{
Internal function that runs language detection using a pre-loaded pipeline.
}
