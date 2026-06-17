% tinyrox says don't edit this manually, but it can't stop you!
\name{serve}
\alias{serve}
\title{Serve whisper over HTTP}
\usage{
serve(port = 4123L, model = "large-v3", device = "cuda", dtype = "auto",
      timeout = 300L, max_body = 100L * 1024L^2, warmup = TRUE)
}
\arguments{
\item{port}{Integer. TCP port to listen on. Default 4123.}

\item{model}{Model name to load and keep resident (the request's
\code{model} field is ignored). Default "large-v3".}

\item{device}{Character. Torch device ("cuda", "cpu", "mps").}

\item{dtype}{Compute dtype ("auto", "float16", "float32").}

\item{timeout}{Integer. Per-connection I/O timeout in seconds (guards
against stalled clients). Default 300.}

\item{max_body}{Integer. Maximum request body size in bytes. Default
100 MB (audio uploads are larger than JSON bodies).}

\item{warmup}{Logical. Transcribe a short bundled clip at startup to
compile the decode step and prime the allocator, so the first client
request isn't slow. Default TRUE.}
}
\value{
Does not return normally; runs until interrupted.
}
\description{
Starts a blocking HTTP server that loads a whisper model once and
answers OpenAI-compatible speech-to-text requests. Intended as a
drop-in for the OpenAI transcription API or a Whisper container: point
an HTTP client (e.g. \pkg{stt.api} via \code{set_stt_base()}) at
\code{http://<host>:<port>} and it serves the same endpoint.
}
\details{
Endpoints:
\itemize{
\item \code{GET /health} - liveness probe, returns
\code{{"status":"ok","model":...}}.
\item \code{POST /v1/audio/transcriptions} - multipart/form-data with
fields \code{file} (required audio upload), \code{language},
\code{response_format} (\code{json} (default), \code{text}, or
\code{verbose_json}), and \code{temperature}. Returns the
transcription. \code{verbose_json} adds \code{segments} with
start/end times.
\item \code{POST /v1/audio/translations} - same, but translates to
English (Whisper's translate task).
}

The server is single-threaded and runs until interrupted. Run it under
a process supervisor (systemd, a container CMD, tmux) for persistence;
an example systemd unit ships with the package:
\code{system.file("whisper.service", package = "whisper")}. It is
designed to sit alongside a chatterbox TTS server as a second always-on
process on the same GPU (each has its own CUDA context).
On CUDA it tunes torch's allocator GC (see \code{\link{whisper_tune_gc}})
before loading and uses the TorchScript greedy decode step.

}
