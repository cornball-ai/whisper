% tinyrox says don't edit this manually, but it can't stop you!
\name{whisper_device}
\alias{whisper_device}
\title{Get Default Device}
\usage{
whisper_device()
}
\value{
torch device object
}
\description{
Returns CUDA device if available, otherwise CPU.
}
\examples{
\donttest{
if (torch::torch_is_installed()) {
  device <- whisper_device()
  device$type
}
}
}
