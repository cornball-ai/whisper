# Test beam search decoding helpers and integration

# --- Unit tests (no model needed) ---

# compression_ratio: repetitive text should have higher ratio
rep_text <- paste(rep("hello world", 50), collapse = " ")
diverse_text <- "The quick brown fox jumps over the lazy dog near the river bank"
expect_true(whisper:::compression_ratio(rep_text) >
  whisper:::compression_ratio(diverse_text),
  info = "repetitive text has higher compression ratio")

# compression_ratio: basic sanity - ratio > 0
expect_true(whisper:::compression_ratio("test") > 0,
  info = "compression ratio is positive")

# --- KV cache tests (require torch) ---
if (requireNamespace("torch", quietly = TRUE) &&
    torch::torch_is_installed()) {

  # expand_kv_cache: verify shapes
  fake_cache <- list(
    list(
      self = list(
        k = torch::torch_randn(c(1L, 4L, 10L, 32L)),
        v = torch::torch_randn(c(1L, 4L, 10L, 32L))
      ),
      cross = list(
        k = torch::torch_randn(c(1L, 4L, 20L, 32L)),
        v = torch::torch_randn(c(1L, 4L, 20L, 32L))
      )
    )
  )

  expanded <- whisper:::expand_kv_cache(fake_cache, 5L)
  expect_equal(expanded[[1]]$self$k$size(1), 5L,
    info = "expand_kv_cache: self key batch dim = beam_size")
  expect_equal(expanded[[1]]$self$v$size(1), 5L,
    info = "expand_kv_cache: self value batch dim = beam_size")
  expect_equal(expanded[[1]]$cross$k$size(1), 5L,
    info = "expand_kv_cache: cross key batch dim = beam_size")
  # Other dims unchanged

  expect_equal(expanded[[1]]$self$k$size(3), 10L,
    info = "expand_kv_cache: seq_len preserved")
  expect_equal(expanded[[1]]$self$k$size(4), 32L,
    info = "expand_kv_cache: head_dim preserved")

  # rearrange_kv_cache: verify values after permutation
  beam3_cache <- list(
    list(
      self = list(
        k = torch::torch_tensor(array(1:24, dim = c(3L, 1L, 2L, 4L)),
          dtype = torch::torch_float()),
        v = torch::torch_tensor(array(25:48, dim = c(3L, 1L, 2L, 4L)),
          dtype = torch::torch_float())
      ),
      cross = NULL
    )
  )

  # Reverse beam order: [3, 2, 1]
  indices <- torch::torch_tensor(c(3L, 2L, 1L), dtype = torch::torch_long())
  rearranged <- whisper:::rearrange_kv_cache(beam3_cache, indices, "cpu")

  # First beam should now have what was beam 3
  orig_beam3 <- as.array(beam3_cache[[1]]$self$k[3, , , ])
  new_beam1 <- as.array(rearranged[[1]]$self$k[1, , , ])
  expect_equal(new_beam1, orig_beam3,
    info = "rearrange_kv_cache: beam 3 moved to position 1")

  # expand_kv_cache with NULL cross
  null_cross_cache <- list(
    list(self = list(k = torch::torch_randn(c(1L, 2L, 5L, 16L)),
                     v = torch::torch_randn(c(1L, 2L, 5L, 16L))),
         cross = NULL)
  )
  expanded2 <- whisper:::expand_kv_cache(null_cross_cache, 3L)
  expect_true(is.null(expanded2[[1]]$cross),
    info = "expand_kv_cache: NULL cross stays NULL")
}

# --- Integration tests (need model) ---
if (tinytest::at_home() && whisper::model_exists("tiny")) {

  audio_file <- system.file("audio", "jfk.mp3", package = "whisper")

  # Beam search produces non-empty text
  result <- whisper::transcribe(audio_file, model = "tiny",
    beam_size = 3L, verbose = FALSE)
  expect_true(nchar(result$text) > 0,
    info = "beam search produces non-empty text")

  # Beam search with timestamps produces valid segments
  result_ts <- whisper::transcribe(audio_file, model = "tiny",
    beam_size = 3L, timestamps = TRUE, verbose = FALSE)
  expect_true(!is.null(result_ts$segments),
    info = "beam search with timestamps produces segments")
  expect_true(nrow(result_ts$segments) > 0,
    info = "beam search with timestamps produces non-empty segments")

  # beam_size=1 matches greedy output
  greedy <- whisper::transcribe(audio_file, model = "tiny",
    beam_size = 1L, verbose = FALSE)
  greedy2 <- whisper::transcribe(audio_file, model = "tiny",
    verbose = FALSE)
  expect_equal(greedy$text, greedy2$text,
    info = "beam_size=1 matches default greedy output")

  # Temperature fallback produces output
  result_fb <- whisper::transcribe(audio_file, model = "tiny",
    temperatures = c(0, 0.4, 0.8), verbose = FALSE)
  expect_true(nchar(result_fb$text) > 0,
    info = "temperature fallback produces non-empty text")
}
