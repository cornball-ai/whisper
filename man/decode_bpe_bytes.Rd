% tinyrox says don't edit this manually, but it can't stop you!
\name{decode_bpe_bytes}
\alias{decode_bpe_bytes}
\title{Decode BPE Bytes Back to Text}
\usage{
decode_bpe_bytes(text)
}
\arguments{
\item{text}{Text with BPE byte tokens}
}
\value{
Decoded UTF-8 text
}
\description{
Reverses the GPT-2 byte-level encoding, converting unicode tokens
back to raw UTF-8 bytes.
}
