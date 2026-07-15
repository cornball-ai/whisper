#!/usr/bin/env Rscript
# Generate torch reference fixtures for the anvl/yunque port. Run with the
# torch whisper installed; writes tools/fixtures/*.rds. Kept separate from the
# anvl verification so torch and anvl never load in the same process.
suppressMessages(library(whisper))

dir.create("tools/fixtures", showWarnings = FALSE, recursive = TRUE)
set.seed(1234)

# --- mel frontend ---
# a synthetic 2 s signal; audio_to_mel pads/trims to 30 s internally
audio <- as.numeric(sin(2 * pi * 220 * (1:32000) / 16000) * 0.3 +
  rnorm(32000, sd = 0.02))
mel <- as.array(audio_to_mel(audio, n_mels = 80L))
saveRDS(list(
  audio = whisper:::pad_or_trim(audio),
  mel_fb = whisper:::load_mel_filterbank(n_mels = 80L),
  mel = mel
), "tools/fixtures/mel.rds")
cat(sprintf("mel fixture: audio %d, mel %s\n", 32000L,
  paste(dim(mel), collapse = "x")))

# --- encoder (real tiny weights) ---
model <- load_whisper_model("tiny")
model$to(device = "cpu", dtype = torch::torch_float()) # fp32 reference
weights_path <- path.expand("~/.cache/whisper/tiny/model.safetensors")
mel_t <- torch::torch_tensor(mel, dtype = torch::torch_float())
enc <- as.array(torch::with_no_grad(model$encoder(mel_t)))
saveRDS(list(mel = mel, enc = enc, weights = weights_path),
  "tools/fixtures/encoder.rds")
cat(sprintf("encoder fixture: enc %s\n", paste(dim(enc), collapse = "x")))

# --- decoder (full forward, no cache) ---
# SOT prompt: <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
tokens <- matrix(c(50258L, 50259L, 50359L, 50363L), nrow = 1L)
logits <- as.array(torch::with_no_grad({
  dec <- model$decoder(
    torch::torch_tensor(tokens, dtype = torch::torch_long()),
    torch::torch_tensor(enc, dtype = torch::torch_float()))
  model$decoder$get_logits(dec$hidden_states)
}))
saveRDS(list(tokens = tokens, xa = enc, logits = logits, weights = weights_path),
  "tools/fixtures/decoder.rds")
cat(sprintf("decoder fixture: logits %s\n", paste(dim(logits), collapse = "x")))
