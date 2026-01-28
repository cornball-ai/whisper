% tinyrox says don't edit this manually, but it can't stop you!
\name{model_exists}
\alias{model_exists}
\title{Check if Model is Downloaded}
\usage{
model_exists(model)
}
\arguments{
\item{model}{Model name}
}
\value{
TRUE if model weights exist locally
}
\description{
Check if Model is Downloaded
}
\examples{
model_exists("tiny")
model_exists("large-v3")
}
