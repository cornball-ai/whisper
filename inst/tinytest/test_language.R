# Test language detection

# Unit tests (no model needed)
langs <- whisper:::whisper_language_table()
expect_equal(length(langs), 99L)
expect_equal(langs[["en"]], 0L)
expect_equal(langs[["su"]], 98L)

# Reverse lookup
expect_equal(whisper:::whisper_lang_from_id(50259L), "en")
expect_equal(whisper:::whisper_lang_from_id(50260L), "zh")
expect_equal(whisper:::whisper_lang_from_id(50357L), "su")

# Round-trip: lang code -> token ID -> lang code
for (code in names(langs)) {
  token_id <- whisper:::whisper_lang_token(code)
  expect_equal(whisper:::whisper_lang_from_id(token_id), code)
}

# Integration tests (need model)
if (at_home() && whisper::model_exists("tiny")) {
  audio_file <- system.file("audio", "jfk.mp3", package = "whisper")

  # detect_language should return English for JFK speech
  result <- whisper::detect_language(audio_file, model = "tiny", verbose = FALSE)
  expect_true(is.list(result))
  expect_equal(result$language, "en")
  expect_true(result$probabilities[["en"]] > 0.5)
  expect_true(length(result$probabilities) == 5L)

  # transcribe with language = NULL should auto-detect
  result <- whisper::transcribe(audio_file, model = "tiny", verbose = FALSE)
  expect_equal(result$language, "en")
  expect_true(nchar(result$text) > 0)
}
