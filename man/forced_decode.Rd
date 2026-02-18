% tinyrox says don't edit this manually, but it can't stop you!
\name{forced_decode}
\alias{forced_decode}
\title{Forced Decode}
\usage{
forced_decode(model, encoder_output, token_ids, device)
}
\arguments{
\item{model}{WhisperModel}

\item{encoder_output}{Encoder hidden states}

\item{token_ids}{Integer vector of all token IDs (including initial)}

\item{device}{Device}
}
\value{
List of cross-attention weight lists (one per content step)
}
\description{
Teacher-forcing decode: feed known token sequence one at a time,
collecting cross-attention weights. Used by beam search when
word_timestamps is needed.
}
