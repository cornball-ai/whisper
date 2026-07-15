#!/usr/bin/env Rscript
# Generate torch reference fixtures for the anvl/yunque port. Run with the
# torch whisper installed. Usage: Rscript tools/gen_fixtures.R [model]
# (default "tiny"). CPU-only. Writes tools/fixtures/<model>/*.rds.
suppressMessages(library(whisper))

model <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(model)) model <- "tiny"
config <- whisper_config(model)
fixdir <- file.path("tools/fixtures", model)
dir.create(fixdir, showWarnings = FALSE, recursive = TRUE)
weights_path <- whisper:::get_weights_path(model)
set.seed(1234)

# --- mel frontend ---
audio <- as.numeric(sin(2 * pi * 220 * (1:32000) / 16000) * 0.3 +
  rnorm(32000, sd = 0.02))
mel <- as.array(audio_to_mel(audio, n_mels = config$n_mels))
saveRDS(list(
  audio = whisper:::pad_or_trim(audio),
  mel_fb = whisper:::load_mel_filterbank(n_mels = config$n_mels),
  mel = mel
), file.path(fixdir, "mel.rds"))
cat(sprintf("[%s] mel %s\n", model, paste(dim(mel), collapse = "x")))

# --- encoder + decoder (real weights, CPU fp32) ---
mdl <- load_whisper_model(model, download = TRUE)
mdl$to(device = "cpu", dtype = torch::torch_float())
mel_t <- torch::torch_tensor(mel, dtype = torch::torch_float())
enc <- as.array(torch::with_no_grad(mdl$encoder(mel_t)))
saveRDS(list(mel = mel, enc = enc, weights = weights_path),
  file.path(fixdir, "encoder.rds"))
cat(sprintf("[%s] encoder %s\n", model, paste(dim(enc), collapse = "x")))

tokens <- matrix(c(50258L, 50259L, 50359L, 50363L), nrow = 1L)
logits <- as.array(torch::with_no_grad({
  dec <- mdl$decoder(torch::torch_tensor(tokens, dtype = torch::torch_long()),
    torch::torch_tensor(enc, dtype = torch::torch_float()))
  mdl$decoder$get_logits(dec$hidden_states)
}))
saveRDS(list(tokens = tokens, xa = enc, logits = logits, weights = weights_path),
  file.path(fixdir, "decoder.rds"))
cat(sprintf("[%s] decoder %s\n", model, paste(dim(logits), collapse = "x")))

# --- end-to-end reference (CPU) ---
jfk <- system.file("audio", "jfk.mp3", package = "whisper")
ref <- transcribe(jfk, model = model, language = "en", jit = FALSE,
  device = "cpu", dtype = "float32", verbose = FALSE)
saveRDS(list(text = ref$text, audio_file = jfk, weights = weights_path),
  file.path(fixdir, "transcribe.rds"))
cat(sprintf("[%s] transcribe: %s\n", model, ref$text))
