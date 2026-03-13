% tinyrox says don't edit this manually, but it can't stop you!
\name{detect_language}
\alias{detect_language}
\title{Language Detection}
\usage{
detect_language(file, model = "tiny", device = "auto", dtype = "auto",
                top_k = 5L, download = TRUE, verbose = TRUE)
}
\arguments{
\item{file}{Path to audio file (WAV, MP3, etc.)}

\item{model}{Model name: "tiny", "base", "small", "medium", "large-v3"}

\item{device}{Device: "auto", "cpu", "cuda"}

\item{dtype}{Data type: "auto", "float16", "float32"}

\item{top_k}{Number of top language probabilities to return (default: 5)}

\item{download}{If TRUE and model not present, prompt to download.}

\item{verbose}{Print loading messages.}
}
\value{
List with \code{language} (two-letter code) and
  \code{probabilities} (named numeric vector of top-k language probs).
}
\description{
Detect the spoken language in an audio file using Whisper.
Detect Language
Identify the spoken language in an audio file. Uses Whisper's decoder
to predict the most likely language token from the first 30 seconds
of audio.
}
\examples{
\donttest{
if (model_exists("tiny")) {
  audio_file <- system.file("audio", "jfk.mp3", package = "whisper")
  result <- detect_language(audio_file)
  result$language
  result$probabilities
}
}
}
