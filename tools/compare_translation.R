#!/usr/bin/env r
# treesitR/bonsaisitter structural audit: numeric constants that drifted
# between a torch reference scope and its anvl/yunque port (drift the parity
# fixtures never exercise). Extend `pairings` as components are ported.
suppressMessages(library(bonsaisitter))

lines_of <- function(path, from, to) {
  paste(readLines(path)[from:to], collapse = "\n")
}

audit <- function(label, reference, port) {
  a <- audit_translation(reference, port, lang = "r")
  cat(sprintf("\n== %s ==\n", label))
  cat("literals in reference not in port:",
    if (length(a$literals_missing)) paste(a$literals_missing, collapse = ", ") else "(none)",
    "\n")
  cat("literals in port not in reference:",
    if (length(a$literals_extra)) paste(a$literals_extra, collapse = ", ") else "(none)",
    "\n")
}

# --- pairings ---
audit("mel frontend",
  paste(lines_of("R/audio.R", 164, 198), lines_of("R/audio.R", 218, 266), sep = "\n"),
  paste(readLines("R/yq_audio.R"), collapse = "\n"))

audit("encoder",
  paste(lines_of("R/encoder.R", 9, 190), lines_of("R/encoder.R", 211, 245), sep = "\n"),
  paste(readLines("R/yq_encoder.R"), collapse = "\n"))

audit("decoder",
  lines_of("R/decoder.R", 11, 208),
  paste(readLines("R/yq_decoder.R"), collapse = "\n"))
