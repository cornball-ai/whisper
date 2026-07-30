% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_activate}
\alias{resident_activate}
\title{Activate a Resident Model (Create the GPU Representation)}
\usage{
resident_activate(res)
}
\arguments{
\item{res}{A `whisper_resident` handle from [resident_load()].}
}
\value{
The handle, invisibly. No-op when already active.
}
\description{
Copies the pinned host weights to the target device in one synchronous
pass (non-blocking per-tensor copies, then a device synchronize). The
transition is transactional: on any failure -- including a partial move
after an out-of-memory error -- every tensor is rebound to its pinned
host copy, the CUDA allocation is released, and the handle returns to
"inactive". A rollback that cannot be verified leaves the handle
"broken", where only [resident_status()] and
[resident_unload()] operate.
}
