% tinyrox says don't edit this manually, but it can't stop you!
\name{rearrange_kv_cache}
\alias{rearrange_kv_cache}
\title{Rearrange KV Cache by Beam Indices}
\usage{
rearrange_kv_cache(kv_cache, beam_indices, device)
}
\arguments{
\item{kv_cache}{List of per-layer KV caches}

\item{beam_indices}{Integer tensor of beam indices (1-indexed)}

\item{device}{Device}
}
\value{
Reordered KV cache
}
\description{
Reorder cached key-value tensors to match new beam ordering.
}
