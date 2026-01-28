% tinyrox says don't edit this manually, but it can't stop you!
\name{whisper_special_tokens}
\alias{whisper_special_tokens}
\title{Special Token IDs}
\usage{
whisper_special_tokens(model = "tiny")
}
\arguments{
\item{model}{Model name (default: "tiny")}
}
\value{
Named list of special token IDs
}
\description{
Get special token IDs for a Whisper model.
Token IDs differ between model variants (e.g., large-v3 has extra language tokens).
}
