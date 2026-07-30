% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_deactivate}
\alias{resident_deactivate}
\title{Deactivate a Resident Model (Destroy the GPU Representation)}
\usage{
resident_deactivate(res)
}
\arguments{
\item{res}{A `whisper_resident` handle.}
}
\value{
The handle, invisibly. No-op when already inactive.
}
\description{
Rebinds every tensor to its pinned host copy, releases the orphaned CUDA
storage, and verifies the restored state tensor-by-tensor. The pinned
weights remain in host memory, so a later [resident_activate()]
is a DMA copy, not a disk reload. Refuses while a transcription is in
flight.
}
