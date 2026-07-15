#!/usr/bin/env r
# Verify the anvl/yunque port against the torch fixtures. Anvl-only process
# (no library(whisper), so torch never loads). CPU-only: anvl defaults to the
# PJRT CpuDevice. Usage: r tools/verify_anvl.R [model] (default "tiny").
suppressMessages({
  library(anvl)
  library(yunque)
})

model <- if (length(argv)) argv[1] else "tiny"
fixdir <- file.path("tools/fixtures", model)

e <- new.env()
for (f in c("config", "suppress", "audio", "tokenizer",
  "yq_audio", "yq_encoder", "yq_decoder", "yq_transcribe")) {
  sys.source(sprintf("R/%s.R", f), envir = e)
}
cfg <- e$whisper_config(model)

reltol <- function(a, b) max(abs(a - b)) / max(abs(b))
ok <- TRUE
report <- function(name, got, ref, tol = 1e-3) {
  r <- reltol(got, ref)
  pass <- identical(dim(got), dim(ref)) && r < tol
  ok <<- ok && pass
  cat(sprintf("[%-10s] %s  reltol %.2e  dim %s\n", name,
    if (pass) "PASS" else "FAIL", r, paste(dim(got), collapse = "x")))
}

# --- mel ---
fix <- readRDS(file.path(fixdir, "mel.rds"))
report("mel", as.array(e$.yq_log_mel(fix$audio, fix$mel_fb)), fix$mel)

# --- encoder ---
fixe <- readRDS(file.path(fixdir, "encoder.rds"))
w <- e$yq_encoder_load_weights(fixe$weights, cfg)
report("encoder", as.array(e$yq_encoder(nv_array(fixe$mel, dtype = "f32"), w)),
  fixe$enc)

# --- decoder (full forward) ---
fixd <- readRDS(file.path(fixdir, "decoder.rds"))
wd <- e$yq_decoder_load_weights(fixd$weights, cfg)
report("decoder", as.array(e$yq_decoder(fixd$tokens,
  nv_array(fixd$xa, dtype = "f32"), wd)), fixd$logits)

# --- end-to-end transcription (KV cache + jit) ---
fixt <- readRDS(file.path(fixdir, "transcribe.rds"))
txt <- e$yq_transcribe(fixt$audio_file, fixt$weights, cfg, model = model)
norm <- function(s) tolower(gsub("[[:punct:][:space:]]+", "", s))
tmatch <- norm(txt) == norm(fixt$text)
ok <- ok && tmatch
cat(sprintf("[%-10s] %s\n  anvl : %s\n  torch: %s\n", "transcribe",
  if (tmatch) "PASS" else "FAIL", txt, fixt$text))

quit(status = if (ok) 0L else 1L)
