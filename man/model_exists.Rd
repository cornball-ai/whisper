% tinyrox says don't edit this manually, but it can't stop you!
\name{model_exists}
\alias{model_exists}
\title{Check if Model is Downloaded}
\usage{
model_exists(model, revision = NULL)
}
\arguments{
\item{model}{Model name}

\item{revision}{Optional exact 40-hex commit to resolve against. With one,
this answers about that snapshot and needs neither a \code{refs/} entry
nor the network.}
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
