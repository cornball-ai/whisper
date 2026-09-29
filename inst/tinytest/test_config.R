# Tests for configuration

# Skip if torch not fully installed (package may not load)
if (!requireNamespace("torch", quietly = TRUE) ||
    !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

# Test config loading
expect_silent(cfg <- whisper_config("tiny"))
expect_equal(cfg$n_mels, 80L)
expect_equal(cfg$n_audio_state, 384L)
expect_equal(cfg$n_audio_layer, 4L)

# Test all model configs exist
for (model in list_whisper_models()) {
  expect_silent(whisper_config(model))
}

# large-v3-turbo: large-v3's encoder, a 4-layer decoder
turbo <- whisper_config("large-v3-turbo")
v3 <- whisper_config("large-v3")
for (k in c("n_mels", "n_audio_state", "n_audio_head", "n_audio_layer",
            "n_vocab", "n_text_state", "n_text_head")) {
  expect_equal(turbo[[k]], v3[[k]], info = k)
}
expect_equal(turbo$n_text_layer, 4L)
expect_true(all(turbo$alignment_heads[, 1] < turbo$n_text_layer))
expect_true(all(turbo$alignment_heads[, 2] < turbo$n_text_head))

# Special tokens follow the vocab: the 51866-token models share one table
expect_identical(whisper:::whisper_special_tokens("large-v3-turbo"),
                 whisper:::whisper_special_tokens("large-v3"))
expect_equal(whisper:::whisper_special_tokens("large-v3-turbo")$timestamp_begin,
             50365L)
expect_equal(whisper:::whisper_special_tokens("medium")$timestamp_begin,
             50364L)

# Test invalid model
expect_error(whisper_config("invalid"))

# Test special tokens
tokens <- whisper:::whisper_special_tokens()
expect_equal(tokens$sot, 50258L)
expect_equal(tokens$eot, 50257L)
expect_equal(tokens$transcribe, 50359L)

# Test language tokens
expect_equal(whisper:::whisper_lang_token("en"), 50259L)
expect_equal(whisper:::whisper_lang_token("es"), 50262L)
expect_error(whisper:::whisper_lang_token("invalid"))

