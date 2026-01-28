% tinyrox says don't edit this manually, but it can't stop you!
\name{get_initial_tokens}
\alias{get_initial_tokens}
\title{Get Initial Decoder Tokens}
\usage{
get_initial_tokens(
  language = "en",
  task = "transcribe",
  model = "tiny",
  timestamps = FALSE
)
}
\arguments{
\item{language}{Two-letter language code or NULL for auto}

\item{task}{"transcribe" or "translate"}

\item{model}{Model name for correct special token IDs}

\item{timestamps}{Whether to include timestamps (internal use)}
}
\value{
Integer vector of initial token IDs
}
\description{
Build the initial token sequence for decoder input.
}
