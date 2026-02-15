# Tests for DTW alignment and word timestamps

# Skip if torch not fully installed
if (!requireNamespace("torch", quietly = TRUE) ||
    !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

# Test medfilt1
# median(1,5)=3, median(1,5,1)=1, median(5,1,1)=1, median(1,1,1)=1, median(1,1)=1
expect_equal(whisper:::medfilt1(c(1, 5, 1, 1, 1), 3L), c(3, 1, 1, 1, 1))
# median(3,1)=2, median(3,1,4)=3, median(1,4,1)=1, median(4,1,5)=4, median(1,5)=3
expect_equal(whisper:::medfilt1(c(3, 1, 4, 1, 5), 3L), c(2, 3, 1, 4, 3))
expect_equal(whisper:::medfilt1(numeric(0), 3L), numeric(0))
expect_equal(whisper:::medfilt1(c(42), 3L), c(42))

# Test dtw_align with known cost matrix
# Simple diagonal cost: cheapest path should follow diagonal
cost <- matrix(10, nrow = 3, ncol = 3)
diag(cost) <- 0
path <- whisper:::dtw_align(cost)

# Path should visit all 3 token positions
expect_true(all(1:3 %in% path[, 1]))
# Path should be monotonically increasing
expect_true(all(diff(path[, 1]) >= 0))
expect_true(all(diff(path[, 2]) >= 0))
# Path starts at (1,1) and ends at (n,m)
expect_equal(path[1, ], c(1L, 1L))
expect_equal(path[nrow(path), ], c(3L, 3L))

# Test dtw_align with rectangular cost matrix
cost2 <- matrix(1, nrow = 2, ncol = 5)
cost2[1, 2] <- 0  # Token 1 maps to frame 2
cost2[2, 4] <- 0  # Token 2 maps to frame 4
path2 <- whisper:::dtw_align(cost2)
expect_equal(path2[1, ], c(1L, 1L))
expect_equal(path2[nrow(path2), ], c(2L, 5L))

# Test group_into_words
if (model_exists("tiny")) {
  tok <- whisper_tokenizer("tiny")

  # Simple case: known token IDs for space-delimited words
  hello_ids <- tok$encode(" hello")
  world_ids <- tok$encode(" world")
  all_ids <- c(hello_ids, world_ids)

  starts <- seq(0, by = 0.5, length.out = length(all_ids))
  ends <- starts + 0.5

  words <- whisper:::group_into_words(all_ids, starts, ends, tok)
  expect_true(is.data.frame(words))
  expect_true(nrow(words) >= 2)
  expect_true("word" %in% names(words))
  expect_true("start" %in% names(words))
  expect_true("end" %in% names(words))
}

# Integration test with word timestamps (requires model + audio)
if (at_home() && model_exists("tiny")) {
  audio_file <- system.file("audio", "jfk.mp3", package = "whisper")
  if (file.exists(audio_file)) {
    result <- transcribe(audio_file, model = "tiny", word_timestamps = TRUE,
      verbose = FALSE)
    expect_true("words" %in% names(result))
    expect_true(is.data.frame(result$words))
    expect_true(nrow(result$words) > 0)
    expect_true("word" %in% names(result$words))
    expect_true("start" %in% names(result$words))
    expect_true("end" %in% names(result$words))
    # Word timestamps should be monotonically non-decreasing
    if (nrow(result$words) > 1) {
      starts <- result$words$start
      expect_true(all(diff(starts) >= -0.01))  # allow tiny float imprecision
    }
    # word_timestamps implies segments are also present
    expect_true("segments" %in% names(result))
  }
}
