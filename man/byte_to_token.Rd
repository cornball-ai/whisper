% tinyrox says don't edit this manually, but it can't stop you!
\name{byte_to_token}
\alias{byte_to_token}
\title{Convert Byte to BPE Token}
\usage{
byte_to_token(byte)
}
\arguments{
\item{byte}{Integer byte value (0-255)}
}
\value{
Character token
}
\description{
GPT-2/Whisper uses a specific byte-to-unicode mapping.
}
