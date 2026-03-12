% tinyrox says don't edit this manually, but it can't stop you!
\name{compute_word_timestamps}
\alias{compute_word_timestamps}
\title{Word-Level Timestamp Alignment}
\usage{
compute_word_timestamps(tokens, cross_attn_weights, tokenizer, config,
                        time_offset = 0, sample_begin = 4L)
}
\arguments{
\item{tokens}{Integer vector of generated token IDs}

\item{cross_attn_weights}{List of cross-attention weight tensors per decode step}

\item{tokenizer}{Whisper tokenizer}

\item{config}{Model configuration}

\item{time_offset}{Time offset in seconds (for chunked audio)}

\item{sample_begin}{Index where content tokens start in generated}
}
\value{
Data frame with word, start, end columns
}
\description{
DTW-based alignment of tokens to audio frames using cross-attention weights.
Compute Word-Level Timestamps
Use cross-attention weights and DTW alignment to assign timestamps
to individual words.
}
