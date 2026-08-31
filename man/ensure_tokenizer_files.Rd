% tinyrox says don't edit this manually, but it can't stop you!
\name{ensure_tokenizer_files}
\alias{ensure_tokenizer_files}
\title{Ensure Tokenizer Files are Downloaded}
\usage{
ensure_tokenizer_files(model, revision = NULL)
}
\arguments{
\item{model}{Model name}

\item{revision}{Optional exact 40-hex commit to resolve against. Threaded
through with the weights: a snapshot bound read-only holds the tokenizer
too, and resolving the weights by commit while resolving the tokenizer by
branch would half-pin the model and fail on the half nobody pinned.}
}
\value{
Path to vocab directory (directory containing vocab.json)
}
\description{
Ensure Tokenizer Files are Downloaded
}
