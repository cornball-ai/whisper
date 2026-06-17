% tinyrox says don't edit this manually, but it can't stop you!
\name{whisper_tune_gc}
\alias{whisper_tune_gc}
\title{Tune torch's CUDA garbage collection for whisper inference}
\usage{
whisper_tune_gc(model = "large-v3", device = "auto", dtype = "auto",
                footprint_gb = NULL)
}
\arguments{
\item{model}{Whisper model name, used to estimate the footprint when
\code{footprint_gb} is NULL.}

\item{device}{Device, as accepted by \code{\link{load_whisper_model}}.}

\item{dtype}{Compute dtype; determines bytes per parameter.}

\item{footprint_gb}{Optional explicit footprint in GB, overriding the
per-model estimate (use for combined multi-model workloads).}
}
\value{
The reserved-rate that was set (invisibly), or NULL when nothing
  was set (non-CUDA device, or the option was already set).
}
\description{
Opt-in performance helper. torch's CUDA allocator invokes R's \code{gc()}
on nearly every allocation once a loaded model occupies more than 20\\% of
GPU memory (its default \code{torch.cuda_allocator_reserved_rate} floor),
which can dominate inference time for the larger whisper models. This raises
the floor to the model's footprint as a fraction of VRAM (clamped to at
least 0.2, so models already under the default are unaffected) and lifts
\code{torch.threshold_call_gc} off its 4 GB default.
}
\details{
Call this \emph{before} \code{\link{load_whisper_model}}: torch reads the
allocator rates once, at lazy CUDA initialization. It is a no-op on
non-CUDA devices and only sets an option that is not already set, so an
explicit \code{options(torch.cuda_allocator_reserved_rate = ...)} always
wins.

\strong{Side effect:} it sets session-global \code{torch.*} options that
persist after the call - deliberately, since torch reads them later. The
package never calls this for you; you invoke it.

For several models resident on one GPU in the \emph{same} R process, pass
their combined size via \code{footprint_gb} so the single shared floor
covers all of them.
}
\examples{
# No-op off CUDA; returns NULL.
whisper_tune_gc("large-v3", device = "cpu")
\dontrun{
# On a GPU, call before loading so torch picks up the rate at CUDA init:
whisper_tune_gc("large-v3", device = "cuda")
model <- load_whisper_model("large-v3", device = "cuda")
}
}
