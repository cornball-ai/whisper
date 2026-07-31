% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_deactivate}
\alias{resident_deactivate}
\title{Deactivate a Resident Model (Destroy the GPU Representation)}
\usage{
resident_deactivate(res, release = TRUE)
}
\arguments{
\item{res}{A `whisper_resident` handle.}

\item{release}{Return the allocator's blocks to the driver (default
TRUE). FALSE keeps them pooled for a fast next activation; the
weights are freed either way, and `gpu_bytes` goes to zero in both
cases (the retained pool is process overhead, attributable to no
model).}
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
\details{
`release` decides who gets the freed VRAM, and it is worth about an
order of magnitude. With `release = TRUE` (the default) the CUDA
caching allocator hands its blocks back to the driver, so the memory is
visible as free to other processes -- but the next activation must
re-acquire every block from the driver, which on a small card measured
~9 ms per tensor (medium fp32, 2.85 GB across 948 tensors: 9.2 s,
0.31 GB/s). With `release = FALSE` the blocks stay in this process's
pool for the next model to reuse, and the same activation took 0.86 s
(3.29 GB/s), matching raw pinned-DMA bandwidth on that card. Use
`FALSE` when one process hosts every model and switches between
them; use the default when a different process needs the card.

}
