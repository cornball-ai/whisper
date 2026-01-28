% tinyrox says don't edit this manually, but it can't stop you!
\name{whisper_dtype}
\alias{whisper_dtype}
\title{Get Default Dtype}
\usage{
whisper_dtype(device = whisper_device())
}
\arguments{
\item{device}{torch device}
}
\value{
torch dtype
}
\description{
Returns float16 on CUDA, float32 on CPU.
}
\examples{
dtype <- whisper_dtype()
dtype
}
