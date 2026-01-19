# Tests for transcribe function

# Skip if torch not fully installed (package may not load)
if (!requireNamespace("torch", quietly = TRUE) ||
    !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

# Test clean_text
expect_equal(whisper:::clean_text("  hello  world  "), "hello world")
expect_equal(whisper:::clean_text("<|en|>hello<|endoftext|>"), "hello")

# Test model listing
models <- list_whisper_models()
expect_true("tiny" %in% models)
expect_true("small" %in% models)
expect_true("large-v3" %in% models)

# Integration test (requires model download - skip if not available)
if (FALSE) {
  # This would be enabled for manual testing
  result <- transcribe("test.wav", model = "tiny")
  expect_true("text" %in% names(result))
  expect_true("language" %in% names(result))
}

