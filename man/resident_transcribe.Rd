% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_transcribe}
\alias{resident_transcribe}
\title{Transcribe with a Resident Model}
\usage{
resident_transcribe(res, file, ...)
}
\arguments{
\item{res}{A `whisper_resident` handle, currently active.}

\item{file}{Path to audio file (WAV, MP3, MP4, ...).}

\item{...}{Passed to the pipeline transcriber: `language`,
`task`, `timestamps`, `word_timestamps`,
`beam_size`, `temperatures`, ... (see [transcribe()]).}
}
\value{
The transcription result, same shape as [transcribe()].
}
\description{
Runs [transcribe()] through the resident handle's pipeline.
Requires the model to be active; marks the handle in-flight for the
duration so a concurrent deactivation is refused.
}
