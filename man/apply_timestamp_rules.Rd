% tinyrox says don't edit this manually, but it can't stop you!
\name{apply_timestamp_rules}
\alias{apply_timestamp_rules}
\title{Apply Timestamp Token Rules}
\usage{
apply_timestamp_rules(logits, generated, special, sample_begin)
}
\arguments{
\item{logits}{Logit tensor (1, vocab) or (vocab)}

\item{generated}{Integer vector of tokens generated so far}

\item{special}{Special token IDs}

\item{sample_begin}{Index where content tokens start in generated}
}
\value{
Modified logits tensor
}
\description{
Enforce Whisper timestamp generation constraints on logits.
}
