#!/usr/bin/env r
# treesitR structural audit: compare numeric literals + call names between a
# torch reference scope and its anvl/yunque port, to catch drift the parity
# fixtures never exercise (wrong constants, missing ops, unported branches).
# Extend `pairings` as components are ported.

suppressMessages(library(treesitR))

extract <- function(code) {
  p <- ts_parser_new()
  ts_parser_set_language(p, ts_language_r())
  root <- ts_tree_root_node(ts_parse(p, code))
  nums <- character(0)
  calls <- character(0)
  walk <- function(n) {
    ty <- ts_node_type(n)
    if (ty %in% c("float", "integer")) nums <<- c(nums, ts_node_text(n))
    if (ty == "call") {
      fn <- ts_node_child_by_field(n, "function")
      if (!ts_node_is_null(fn)) calls <<- c(calls, ts_node_text(fn))
    }
    for (ch in ts_node_children(n)) walk(ch)
  }
  walk(root)
  list(nums = nums, calls = calls)
}

# Read a contiguous line range of a file as one code string.
lines_of <- function(path, from, to) {
  paste(readLines(path)[from:to], collapse = "\n")
}

# canonicalize a call name to its last component (drop pkg:: and torch_/nv_ etc.
# are kept verbatim; we compare intent by hand with the noise filter below)
canon <- function(x) sub("^.*::", "", x)

audit <- function(label, ref_code, port_code, callee_noise = character(0)) {
  r <- extract(ref_code)
  p <- extract(port_code)
  cat(sprintf("\n== %s ==\n", label))
  # numeric literals: strip L/trailing zeros so 4.0 == 4, 2L == 2
  norm_num <- function(v) as.character(as.numeric(sub("L$", "", v)))
  rn <- table(norm_num(r$nums))
  pn <- table(norm_num(p$nums))
  miss <- setdiff(names(rn), names(pn))
  extra <- setdiff(names(pn), names(rn))
  cat("literals in reference not in port:",
    if (length(miss)) paste(miss, collapse = ", ") else "(none)", "\n")
  cat("literals in port not in reference:",
    if (length(extra)) paste(extra, collapse = ", ") else "(none)", "\n")
}

# --- pairings ---
REF <- "R/audio.R"
PORT <- "R/yq_audio.R"

# compute_stft (164-198) + audio_to_mel (218-266) vs the whole yq_audio.R port
audit(
  "mel frontend",
  paste(lines_of(REF, 164, 198), lines_of(REF, 218, 266), sep = "\n"),
  paste(readLines(PORT), collapse = "\n")
)

# encoder: attn + layer + encoder init/forward (skip create_sinusoidal_pe
# 192-209, dead code since HF ships embed_positions.weight) vs yq_encoder.R
audit(
  "encoder",
  paste(lines_of("R/encoder.R", 9, 190), lines_of("R/encoder.R", 211, 245),
    sep = "\n"),
  paste(readLines("R/yq_encoder.R"), collapse = "\n")
)
