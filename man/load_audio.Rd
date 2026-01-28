% tinyrox says don't edit this manually, but it can't stop you!
\name{load_audio}
\alias{load_audio}
\title{Load and Preprocess Audio}
\usage{
load_audio(file)
}
\arguments{
\item{file}{Path to audio file (WAV, MP3, etc.)}
}
\value{
Numeric vector of audio samples normalized to -1 to 1 range
}
\description{
Load audio from file, convert to mono, resample to 16kHz.
}
\examples{
# Load included sample audio
audio_file <- system.file("audio", "jfk.mp3", package = "whisper")
samples <- load_audio(audio_file)
length(samples)
range(samples)
}
