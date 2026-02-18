% tinyrox says don't edit this manually, but it can't stop you!
\name{build_byte_decoder}
\alias{build_byte_decoder}
\title{Build Reverse Byte Decoder}
\usage{
build_byte_decoder()
}
\value{
Named character vector mapping unicode codepoint (as string) to
  raw byte value
}
\description{
Inverts the GPT-2 byte-to-unicode mapping used by byte_to_token().
Cached after first call.
}
