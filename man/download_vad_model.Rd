% tinyrox says don't edit this manually, but it can't stop you!
\name{download_vad_model}
\alias{download_vad_model}
\title{Download the Silero VAD Model}
\usage{
download_vad_model(force = FALSE)
}
\arguments{
\item{force}{Re-download even if the model is already cached.}
}
\value{
Path to the cached model file, invisibly.
}
\description{
Downloads the Silero voice activity detection model (~2 MB, MIT
license) used by \code{\link{whisper_stream}} for endpointing. Follows
the same consent rules as \code{\link{download_whisper_model}}: asks in
interactive sessions, and in non-interactive sessions requires
\code{options(whisper.consent = TRUE)}.
}
\examples{
\dontrun{
download_vad_model()
}
}
