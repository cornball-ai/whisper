#!/usr/bin/env r
# Verify the anvl/yunque port against the torch fixtures. Anvl-only process:
# source the port's R files directly (no library(whisper), so torch never
# loads and can't clash with anvl on CUDA).
suppressMessages({
  library(anvl)
  library(yunque)
})

e <- new.env()
sys.source("R/config.R", envir = e) # whisper_config
sys.source("R/audio.R", envir = e) # constants + host helpers (torch untouched)
sys.source("R/yq_audio.R", envir = e)
sys.source("R/suppress.R", envir = e) # suppress/blank token id builders
sys.source("R/tokenizer.R", envir = e) # pure-R BPE + suppress + prompt
sys.source("R/yq_encoder.R", envir = e)
sys.source("R/yq_decoder.R", envir = e)
sys.source("R/yq_transcribe.R", envir = e)

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

# --- encoder ---
fixe <- readRDS("tools/fixtures/encoder.rds")
w <- e$yq_encoder_load_weights(fixe$weights, e$whisper_config("tiny"))
enc <- as.array(e$yq_encoder(anvl::nv_array(fixe$mel, dtype = "f32"), w))
report("encoder", enc, fixe$enc)

# --- decoder (full forward) ---
fixd <- readRDS("tools/fixtures/decoder.rds")
wd <- e$yq_decoder_load_weights(fixd$weights, e$whisper_config("tiny"))
logits <- as.array(e$yq_decoder(fixd$tokens,
  anvl::nv_array(fixd$xa, dtype = "f32"), wd))
report("decoder", logits, fixd$logits)

# --- end-to-end transcription (text) ---
fixt <- readRDS("tools/fixtures/transcribe.rds")
txt <- e$yq_transcribe(fixt$audio_file, fixt$weights, e$whisper_config("tiny"))
norm <- function(s) tolower(gsub("[[:punct:][:space:]]+", "", s))
tmatch <- norm(txt) == norm(fixt$text)
ok <- ok && tmatch
cat(sprintf("[%-14s] %s\n  anvl : %s\n  torch: %s\n", "transcribe",
  if (tmatch) "PASS" else "FAIL", txt, fixt$text))

quit(status = if (ok) 0L else 1L)
