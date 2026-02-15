% tinyrox says don't edit this manually, but it can't stop you!
\name{group_into_words}
\alias{group_into_words}
\title{Group Subword Tokens into Words}
\usage{
group_into_words(token_ids, starts, ends, tokenizer)
}
\arguments{
\item{token_ids}{Integer vector of text token IDs}

\item{starts}{Numeric vector of token start times}

\item{ends}{Numeric vector of token end times}

\item{tokenizer}{Whisper tokenizer}
}
\value{
Data frame with word, start, end columns
}
\description{
Merge BPE subword tokens into whole words with timestamps.
}
