% tinyrox says don't edit this manually, but it can't stop you!
\name{tokenizer_encode}
\alias{tokenizer_encode}
\title{Encode Text to Token IDs}
\usage{
tokenizer_encode(text, vocab, merge_ranks)
}
\arguments{
\item{text}{Character string to encode}

\item{vocab}{Vocabulary mapping (token -> id)}

\item{merge_ranks}{Merge ranking for BPE}
}
\value{
Integer vector of token IDs
}
\description{
Encode Text to Token IDs
}
