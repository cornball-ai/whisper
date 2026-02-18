% tinyrox says don't edit this manually, but it can't stop you!
\name{expand_kv_cache}
\alias{expand_kv_cache}
\title{Expand KV Cache for Beam Search}
\usage{
expand_kv_cache(kv_cache, beam_size)
}
\arguments{
\item{kv_cache}{List of per-layer KV caches (batch=1)}

\item{beam_size}{Number of beams}
}
\value{
Expanded KV cache (batch=beam_size)
}
\description{
Replicate batch=1 KV cache to batch=beam_size.
}
