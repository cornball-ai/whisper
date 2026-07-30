% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_status}
\alias{resident_status}
\title{Status of a Resident Model}
\usage{
resident_status(res)
}
\arguments{
\item{res}{A `whisper_resident` handle.}
}
\value{
A list: model, state, in_flight, device, dtype, pinned_bytes,
  gpu_bytes (logical sums over the tensor manifest, not allocator
  statistics), identity (weights sha256, HF repo/revision, resolved
  dtype), path (diagnostic), last_error.
}
\description{
Callable in every state, including "broken" and "unloaded".
}
