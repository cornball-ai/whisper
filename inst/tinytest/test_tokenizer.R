# Tests for tokenizer

# Skip if torch not fully installed (package may not load)
if (!requireNamespace("torch", quietly = TRUE) ||
    !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

# Test initial tokens generation (default: no timestamps)
tokens_no_ts <- whisper:::get_initial_tokens("en", "transcribe")
expect_true(50258L %in% tokens_no_ts) # sot
expect_true(50259L %in% tokens_no_ts) # en
expect_true(50359L %in% tokens_no_ts) # transcribe
expect_true(50363L %in% tokens_no_ts) # no_timestamps

# Test with timestamps enabled
tokens_ts <- whisper:::get_initial_tokens("en", "transcribe", timestamps = TRUE)
expect_false(50363L %in% tokens_ts) # no_timestamps should NOT be present

# Test timestamp token detection
expect_false(whisper:::is_timestamp_token(50257L)) # eot
expect_true(whisper:::is_timestamp_token(50364L)) # first timestamp

# Test timestamp decoding
expect_equal(whisper:::decode_timestamp(50364L), 0)
expect_equal(whisper:::decode_timestamp(50365L), 0.02)
expect_equal(whisper:::decode_timestamp(50414L), 1.0) # 50 * 0.02

# Test byte_to_token / decode_bpe_bytes round-trip
# ASCII
ascii_text <- "hello world"
ascii_bpe <- paste(sapply(charToRaw(ascii_text), function(b) {
  whisper:::byte_to_token(as.integer(b))
}), collapse = "")
expect_equal(whisper:::decode_bpe_bytes(ascii_bpe), ascii_text)

# Non-ASCII (accented Latin)
utf8_text <- "caf\u00e9"
utf8_bpe <- paste(sapply(charToRaw(utf8_text), function(b) {
  whisper:::byte_to_token(as.integer(b))
}), collapse = "")
expect_equal(whisper:::decode_bpe_bytes(utf8_bpe), utf8_text)

# CJK characters
cjk_text <- "\u4e16\u754c"
cjk_bpe <- paste(sapply(charToRaw(cjk_text), function(b) {
  whisper:::byte_to_token(as.integer(b))
}), collapse = "")
expect_equal(whisper:::decode_bpe_bytes(cjk_bpe), cjk_text)

# Empty string
expect_equal(whisper:::decode_bpe_bytes(""), "")

# Regression (0.3.0.7): tokenizer_encode() must not crash when vocab.json omits
# the <|endoftext|> key, as large-v3's does. Unmatched tokens fall back to the
# supplied eot id, and the result is always an integer vector, never a list
# (the bug returned a list, which as.integer() rejected).
empty_merges <- setNames(integer(0), character(0))
ids_fb <- whisper:::tokenizer_encode("ab", list(), empty_merges, eot_fallback = 99L)
expect_true(is.integer(ids_fb))
expect_equal(ids_fb, c(99L, 99L))                    # no key, no match -> eot fallback
expect_equal(                                        # known token still resolves
  whisper:::tokenizer_encode("a", list("a" = 5L), empty_merges, eot_fallback = 99L),
  5L)

