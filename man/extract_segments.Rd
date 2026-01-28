% tinyrox says don't edit this manually, but it can't stop you!
\name{extract_segments}
\alias{extract_segments}
\title{Extract Segments with Timestamps}
\usage{
extract_segments(tokens, tokenizer, time_offset = 0)
}
\arguments{
\item{tokens}{Token IDs}

\item{tokenizer}{Tokenizer}

\item{time_offset}{Offset in seconds for chunk processing}
}
\value{
Data frame with start, end, text
}
\description{
Extract Segments with Timestamps
}
