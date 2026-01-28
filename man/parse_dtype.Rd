% tinyrox says don't edit this manually, but it can't stop you!
\name{parse_dtype}
\alias{parse_dtype}
\title{Parse Dtype Argument}
\usage{
parse_dtype(dtype = "auto", device = whisper_device())
}
\arguments{
\item{dtype}{Character or torch dtype. "auto" uses float16 on GPU, float32 on CPU.}

\item{device}{torch device (used for auto selection)}
}
\value{
torch dtype
}
\description{
Parse Dtype Argument
}
