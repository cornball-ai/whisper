# Tests for tokenizer

# Test initial tokens generation
tokens <- whisper:::get_initial_tokens("en", "transcribe", timestamps = TRUE)
expect_true(50258L %in% tokens) # sot
expect_true(50259L %in% tokens) # en
expect_true(50359L %in% tokens) # transcribe

tokens_no_ts <- whisper:::get_initial_tokens("en", "transcribe", timestamps = FALSE)
expect_true(50363L %in% tokens_no_ts) # no_timestamps

# Test timestamp token detection
expect_false(whisper:::is_timestamp_token(50257L)) # eot
expect_true(whisper:::is_timestamp_token(50364L)) # first timestamp

# Test timestamp decoding
expect_equal(whisper:::decode_timestamp(50364L), 0)
expect_equal(whisper:::decode_timestamp(50365L), 0.02)
expect_equal(whisper:::decode_timestamp(50414L), 1.0) # 50 * 0.02

