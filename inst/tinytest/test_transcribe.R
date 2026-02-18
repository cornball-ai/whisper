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

# Test extract_segments with synthetic token sequences
# Use tiny model token IDs (timestamp_begin = 50364)
special <- whisper:::whisper_special_tokens("tiny")
ts_begin <- special$timestamp_begin

# Build a synthetic token sequence: <|0.00|> hello world <|2.50|> <|2.50|> foo <|5.00|>
# Token IDs: ts_begin + 0, some text tokens, ts_begin + 125, ts_begin + 125, text, ts_begin + 250
if (model_exists("tiny")) {
  tok <- whisper_tokenizer("tiny")

  # Encode "hello" to get real token IDs
  hello_ids <- tok$encode("hello")
  foo_ids <- tok$encode("foo")

  synthetic_tokens <- c(
    special$sot, special$lang_en, special$transcribe,  # prompt tokens
    ts_begin,            # <|0.00|>
    hello_ids,           # "hello"
    ts_begin + 125L,     # <|2.50|>
    ts_begin + 125L,     # <|2.50|>
    foo_ids,             # "foo"
    ts_begin + 250L      # <|5.00|>
  )

  segments <- whisper:::extract_segments(synthetic_tokens, tok, time_offset = 0)
  expect_true(is.data.frame(segments))
  expect_true(nrow(segments) >= 2)
  expect_equal(segments$start[1], 0)
  expect_equal(segments$end[1], 2.5)
  expect_equal(segments$start[2], 2.5)
  expect_equal(segments$end[2], 5.0)

  # Test with time_offset
  segments2 <- whisper:::extract_segments(synthetic_tokens, tok, time_offset = 10)
  expect_equal(segments2$start[1], 10)
  expect_equal(segments2$end[1], 12.5)
}

# Test apply_timestamp_rules with synthetic logits
n_vocab <- special$timestamp_begin + 1501L  # full vocab
logits_base <- torch::torch_zeros(1, n_vocab)

# Rule 1: First content token must be <|0.00|>
logits1 <- logits_base$clone()
logits1[1, ts_begin + 1] <- 5.0  # <|0.00|> should be allowed (1-indexed)
logits1[1, 100] <- 10.0  # text token should be suppressed
result1 <- whisper:::apply_timestamp_rules(logits1, integer(4), special, 4L)
# Text tokens should be suppressed (-Inf)
expect_true(as.numeric(result1[1, 100]$item()) == -Inf)
# <|0.00|> should be allowed
expect_true(as.numeric(result1[1, ts_begin + 1]$item()) > -Inf)

# Integration test with timestamps (requires model + audio)
if (at_home() && model_exists("tiny")) {
  audio_file <- system.file("audio", "jfk.mp3", package = "whisper")
  if (file.exists(audio_file)) {
    result <- transcribe(audio_file, model = "tiny", timestamps = TRUE,
      verbose = FALSE)
    expect_true("segments" %in% names(result))
    expect_true(is.data.frame(result$segments))
    expect_true(nrow(result$segments) > 0)
    expect_true("start" %in% names(result$segments))
    expect_true("end" %in% names(result$segments))
    expect_true("text" %in% names(result$segments))
    # Segments should have monotonic start times
    if (nrow(result$segments) > 1) {
      starts <- result$segments$start
      expect_true(all(diff(starts) >= 0))
    }
  }
}

