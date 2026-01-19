# Tests for configuration

# Test config loading
expect_silent(cfg <- whisper_config("tiny"))
expect_equal(cfg$n_mels, 80L)
expect_equal(cfg$n_audio_state, 384L)
expect_equal(cfg$n_audio_layer, 4L)

# Test all model configs exist
for (model in c("tiny", "base", "small", "medium", "large-v3")) {
  expect_silent(whisper_config(model))
}

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

