#!/usr/bin/env r
# Verify the anvl/yunque port against the torch fixtures. Anvl-only process:
# source the port's R files directly (no library(whisper), so torch never
# loads and can't clash with anvl on CUDA).
suppressMessages({
  library(anvl)
  library(yunque)
})

e <- new.env()
sys.source("R/audio.R", envir = e) # constants + host helpers (torch untouched)
sys.source("R/yq_audio.R", envir = e)

reltol <- function(a, b) max(abs(a - b)) / max(abs(b))
ok <- TRUE
report <- function(name, got, ref, tol = 1e-3) {
  r <- reltol(got, ref)
  pass <- identical(dim(got), dim(ref)) && r < tol
  ok <<- ok && pass
  cat(sprintf("[%-14s] %s  reltol %.2e  dim %s\n", name,
    if (pass) "PASS" else "FAIL", r, paste(dim(got), collapse = "x")))
}

# --- mel frontend ---
fix <- readRDS("tools/fixtures/mel.rds")
got <- as.array(e$.yq_log_mel(fix$audio, fix$mel_fb))
report("mel", got, fix$mel)

quit(status = if (ok) 0L else 1L)
