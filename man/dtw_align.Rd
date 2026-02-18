% tinyrox says don't edit this manually, but it can't stop you!
\name{dtw_align}
\alias{dtw_align}
\title{DTW Alignment}
\usage{
dtw_align(cost)
}
\arguments{
\item{cost}{Numeric matrix (n_tokens x n_frames)}
}
\value{
Integer matrix with 2 columns (token_idx, frame_idx), 1-indexed
}
\description{
Standard dynamic time warping on a cost matrix.
}
