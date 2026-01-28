% tinyrox says don't edit this manually, but it can't stop you!
\name{whisper_decoder}
\alias{whisper_decoder}
\title{Text Decoder}
\usage{
whisper_decoder(n_vocab, n_ctx, n_state, n_head, n_layer)
}
\arguments{
\item{n_vocab}{Vocabulary size}

\item{n_ctx}{Maximum context length}

\item{n_state}{Hidden dimension}

\item{n_head}{Number of attention heads}

\item{n_layer}{Number of transformer layers}
}
\description{
Full Whisper decoder: token embedding + positional embedding + transformer layers.
}
