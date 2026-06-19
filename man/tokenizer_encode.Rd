% tinyrox says don't edit this manually, but it can't stop you!
\name{tokenizer_encode}
\alias{tokenizer_encode}
\title{Encode Text to Token IDs}
\usage{
tokenizer_encode(text, vocab, merge_ranks, eot_fallback = NULL)
}
\arguments{
\item{text}{Character string to encode}

\item{vocab}{Vocabulary mapping (token -> id)}

\item{merge_ranks}{Merge ranking for BPE}

\item{eot_fallback}{End-of-text id for any token not found in \code{vocab}.
Like the Python reference (tiktoken), \code{<|endoftext|>} is a special
token kept out of the BPE vocab; some vocab.json files (large-v3) omit it
entirely, so the id is supplied from the special-token table.}
}
\value{
Integer vector of token IDs
}
\description{
Encode Text to Token IDs
}
