% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_unload}
\alias{resident_unload}
\title{Unload a Resident Model (Drop Pinned and GPU State)}
\usage{
resident_unload(res)
}
\arguments{
\item{res}{A `whisper_resident` handle.}
}
\value{
The handle, invisibly.
}
\description{
Releases both representations: any GPU tensors and the pinned host
copies. Permitted from every state except mid-flight -- including
"broken", where it is the recovery path (dropping the module and the
pinned list releases the storage that could not be verified). The handle
becomes "unloaded", which is terminal.
}
