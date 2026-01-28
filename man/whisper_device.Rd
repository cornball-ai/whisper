% tinyrox says don't edit this manually, but it can't stop you!
\name{whisper_device}
\alias{whisper_device}
\title{Device and Dtype Management}
\usage{
whisper_device()
}
\value{
torch device object
}
\description{
Utilities for managing torch devices and data types.
Get Default Device
Returns CUDA device if available, otherwise CPU.
}
\examples{
device <- whisper_device()
device$type
}
