% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_load}
\alias{resident_load}
\title{Load a Model as a Resident (Pinned Host Weights)}
\usage{
resident_load(model = "tiny", device = "cuda", dtype = "auto", download = TRUE,
              verbose = TRUE)
}
\arguments{
\item{model}{Model name: "tiny", "base", "small", "medium", "large-v3"}

\item{device}{Target CUDA device for activation (default "cuda").
Residency requires CUDA; pinned host memory exists to feed it. A bare
"cuda" is resolved to the current device's explicit index at load
time, and the handle stays bound to that exact device (`"cuda:N"`)
for every later transition and synchronize.}

\item{dtype}{"auto" (default), "float16", or "float32"; resolved against
`device`.}

\item{download}{Download the model if not cached (default TRUE).}

\item{verbose}{Print progress messages.}
}
\value{
A `whisper_resident` handle (an environment). Fields of
  interest via [resident_status()]: state, byte counts, and a
  content identity (weights sha256, HF repo and snapshot revision,
  resolved dtype).
}
\description{
Loads a Whisper model and retains its canonical weights as page-locked
(pinned) CPU tensors, so the GPU representation can be created and
destroyed repeatedly without reloading from disk.
[resident_activate()] copies the weights to the GPU (a fast DMA
transfer from pinned memory); [resident_deactivate()] frees the
GPU copy and repoints the model at the pinned host storage. The handle
starts inactive.
}
\details{
The dtype is resolved against the *target* device with the same rules as
[whisper_pipeline()]: float16 on CUDA, except GPUs with broken
fp16 (GTX 16-series), which get float32. The pinned copies are stored at
the resolved dtype, so activation moves exactly the bytes inference needs.

}
\examples{
\donttest{
if (torch::torch_is_installed() && torch::cuda_is_available() &&
  model_exists("tiny")) {
  res <- resident_load("tiny")
  resident_activate(res)
  audio <- system.file("audio", "jfk.mp3", package = "whisper")
  resident_transcribe(res, audio, timestamps = TRUE)
  resident_deactivate(res) # VRAM freed, weights stay pinned in RAM
  resident_activate(res) # fast: DMA copy, no disk
  resident_unload(res)
}
}
}
