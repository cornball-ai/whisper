% tinyrox says don't edit this manually, but it can't stop you!
\name{whisper_config}
\alias{whisper_config}
\title{Whisper Model Configurations}
\usage{
whisper_config(model = "tiny")
}
\arguments{
\item{model}{Character. Model name: "tiny", "base", "small", "medium", "large-v3"}
}
\value{
List with model configuration parameters
}
\description{
Get configuration for a Whisper model variant.
}
\examples{
# Get tiny model configuration
cfg <- whisper_config("tiny")
cfg$n_mels
cfg$n_audio_layer

# Compare model sizes
whisper_config("tiny")$n_text_layer
whisper_config("large-v3")$n_text_layer
}
