% tinyrox says don't edit this manually, but it can't stop you!
\name{whisper_stream}
\alias{whisper_stream}
\title{Open a Streaming Transcription Session}
\usage{
whisper_stream(
  pipe,
  language = NULL,
  task = "transcribe",
  vad = c("silero", "energy"),
  onset_prob = 0.5,
  offset_prob = 0.35,
  min_speech_ms = 250,
  endpoint_silence_ms = 700,
  punct_silence_ms = 500,
  midsentence_silence_ms = 1000,
  pre_roll_ms = 320,
  decode_interval_ms = 1000,
  max_utterance_s = 28,
  final_temperatures = c(0, 0.2, 0.4, 0.6, 0.8, 1),
  final_beam_size = 1L,
  download = TRUE,
  verbose = FALSE
)
}
\arguments{
\item{pipe}{A \code{\link{whisper_pipeline}} object.}

\item{language}{Language code (e.g. "en"), or NULL (default) to detect
once on the first utterance and pin for the session.}

\item{task}{"transcribe" or "translate".}

\item{vad}{"silero" (default) or "energy" (no model download; markedly
weaker endpointing, documented for offline use).}

\item{onset_prob}{Speech probability that opens a speech run.}

\item{offset_prob}{Hysteresis threshold that extends one.}

\item{min_speech_ms}{Shorter blips never open a turn.}

\item{endpoint_silence_ms}{Base trailing-silence target that ends a
turn.}

\item{punct_silence_ms}{Silence target when the transcript tail ends a
sentence.}

\item{midsentence_silence_ms}{Silence target mid-sentence.}

\item{pre_roll_ms}{Audio kept from before the detected onset.}

\item{decode_interval_ms}{New audio per incremental decode.}

\item{max_utterance_s}{Forced-endpoint cap on a single turn.}

\item{final_temperatures}{Temperature ladder for the flush decode.}

\item{final_beam_size}{Beam size for the flush decode.}

\item{download}{Download the VAD model if missing (consent rules as in
\code{\link{download_vad_model}}).}

\item{verbose}{Message on turn events.}
}
\value{
A \code{whisper_stream} object.
}
\description{
Creates a live transcription session over a loaded
\code{\link{whisper_pipeline}}: feed it 16 kHz mono PCM as it arrives
and receive transcript events as they are produced, including the
endpointing judgment that the speaker's turn is over.
}
\details{
The returned object has three methods:
\itemize{
\item \code{$feed(pcm)} - consume the next audio chunk. \code{pcm}
is a raw vector (signed 16-bit little-endian PCM), an integer
vector of 16-bit sample values, or a numeric vector of float
samples in [-1, 1]. Chunks can be any length; timing is derived
from the cumulative sample count, never from chunk sizes. Returns
a list of zero or more events.
\item \code{$end()} - end of stream: flushes any open turn and
returns its final events.
\item \code{$close()} - drop buffers and VAD state.
}

Events are lists with a \code{type} field:
\itemize{
\item \code{list(type = "transcript", text =, stable =)} - a stable
transcript APPENDS: the turn's text is the concatenation of stable
events exactly as sent, joining whitespace included. A provisional
transcript (\code{stable = FALSE}) REPLACES the previous
provisional one: it is the whole uncommitted tail.
\item \code{list(type = "speech_ended", audio_offset_ms =)} - the
turn is over. Everything that will ever be stable for the turn is
emitted before this event; the offset points at the end of the
last speech frame in the stream.
}

Endpointing runs on the Silero VAD model (downloaded on first use, ~2
MB, see \code{\link{download_vad_model}}) with an adaptive silence
timer: a tail that reads as a finished sentence endpoints after
\code{punct_silence_ms}, a mid-sentence pause waits
\code{midsentence_silence_ms}, and \code{endpoint_silence_ms} applies
while there is no transcript yet. Incremental transcription commits
what two consecutive decodes agree on (LocalAgreement-2), with
committed tokens forced as the decoder prefix so stable text is never
contradicted.

A turn longer than \code{max_utterance_s} is flushed as a forced
endpoint (the turn splits). One live session per process; decoding
happens inline in \code{$feed()}.

}
\examples{
\dontrun{
pipe <- whisper_pipeline("tiny")
stream <- whisper_stream(pipe, language = "en")
audio <- whisper:::load_audio(
  system.file("audio", "jfk.mp3", package = "whisper"))
for (chunk in split(audio, ceiling(seq_along(audio) / 4000))) {
  for (ev in stream$feed(chunk)) str(ev)
}
for (ev in stream$end()) str(ev)
}
}
