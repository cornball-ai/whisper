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
Returns float16 on CUDA, float32 on CPU. Exception: the GTX 16-series
(TU116/TU117, e.g. GTX 1660/1650) computes float16 incorrectly and
produces NaN output, so float32 is used on those cards. Pass an explicit
\code{dtype = "float16"} to override.
}
\examples{
\donttest{
if (torch::torch_is_installed()) {
  dtype <- whisper_dtype()
  dtype
}
}
}
