#' Word-Level Timestamp Alignment
#'
#' DTW-based alignment of tokens to audio frames using cross-attention weights.

#' Compute Word-Level Timestamps
#'
#' Use cross-attention weights and DTW alignment to assign timestamps
#' to individual words.
#'
#' @param tokens Integer vector of generated token IDs
#' @param cross_attn_weights List of cross-attention weight tensors per decode step
#' @param tokenizer Whisper tokenizer
#' @param config Model configuration
#' @param time_offset Time offset in seconds (for chunked audio)
#' @param sample_begin Index where content tokens start in generated
#' @return Data frame with word, start, end columns
compute_word_timestamps <- function(
  tokens,
  cross_attn_weights,
  tokenizer,
  config,
  time_offset = 0,
  sample_begin = 4L
) {
  if (length(cross_attn_weights) == 0) {
    return(data.frame(word = character(0), start = numeric(0), end = numeric(0)))
  }

  special <- whisper_special_tokens(config$model_name)

  # Content tokens only (after initial prompt tokens)
  content_tokens <- tokens[seq_len(length(tokens)) > sample_begin]

  # Filter out timestamp tokens for word alignment
  text_mask <- content_tokens < special$timestamp_begin
  if (sum(text_mask) == 0) {
    return(data.frame(word = character(0), start = numeric(0), end = numeric(0)))
  }

  # Get alignment heads for this model
  alignment_heads <- config$alignment_heads
  if (is.null(alignment_heads)) {
    # Fallback: use all heads from last half of layers
    n_layer <- config$n_text_layer
    n_head <- config$n_text_head
    half <- n_layer %/% 2L
    layers <- seq(half, n_layer - 1L)
    heads <- seq(0L, n_head - 1L)
    alignment_heads <- as.matrix(expand.grid(layer = layers, head = heads))
  }

  # Build attention matrix: average over alignment heads and decode steps
  # Each element of cross_attn_weights is a list of per-layer tensors
  # Each tensor has shape (batch, n_head, 1, n_audio_ctx)
  n_steps <- length(cross_attn_weights)
  n_audio_ctx <- config$n_audio_ctx

  # Stack attention from alignment heads across all steps
  # Result: (n_steps, n_audio_ctx) averaged over alignment heads
  attn_matrix <- matrix(0, nrow = n_steps, ncol = n_audio_ctx)

  for (step in seq_len(n_steps)) {
    step_weights <- cross_attn_weights[[step]]
    n_heads_used <- 0

    for (h in seq_len(nrow(alignment_heads))) {
      layer_idx <- alignment_heads[h, 1] + 1L  # 0-indexed to 1-indexed
      head_idx <- alignment_heads[h, 2] + 1L

      if (layer_idx <= length(step_weights) && !is.null(step_weights[[layer_idx]])) {
        # step_weights[[layer_idx]] is (batch, n_head, seq_len, src_len)
        w <- step_weights[[layer_idx]]
        # Extract specific head, last query position
        head_attn <- as.array(w[1, head_idx, w$size(3), ]$cpu())
        attn_matrix[step, ] <- attn_matrix[step, ] + head_attn
        n_heads_used <- n_heads_used + 1L
      }
    }

    if (n_heads_used > 0) {
      attn_matrix[step, ] <- attn_matrix[step, ] / n_heads_used
    }
  }

  # Determine audio frame range from timestamp tokens (if present)
  # Find the last timestamp token to cap the attention matrix
  max_frame <- n_audio_ctx
  for (j in rev(seq_along(content_tokens))) {
    if (content_tokens[j] >= special$timestamp_begin) {
      ts_seconds <- (content_tokens[j] - special$timestamp_begin) * 0.02
      max_frame <- min(n_audio_ctx, max(1L, as.integer(ts_seconds / 0.02)))
      break
    }
  }

  # Keep only text token rows (not timestamp tokens)
  text_indices <- which(text_mask)
  if (length(text_indices) == 0) {
    return(data.frame(word = character(0), start = numeric(0), end = numeric(0)))
  }
  text_attn <- attn_matrix[text_indices, 1:max_frame, drop = FALSE]

  # Apply median filter along time axis for smoothing
  text_attn <- apply(text_attn, 1, function(row) medfilt1(row, 7L))
  text_attn <- t(text_attn)

  # Convert to cost matrix for DTW: -log(attn + eps)
  cost <- -log(text_attn + 1e-10)

  # Run DTW alignment
  path <- dtw_align(cost)

  # Map path to per-token frame ranges
  text_token_ids <- content_tokens[text_indices]
  n_text <- length(text_token_ids)
  token_frames <- vector("list", n_text)
  for (k in seq_len(n_text)) {
    token_frames[[k]] <- integer(0)
  }

  for (p in seq_len(nrow(path))) {
    tok_idx <- path[p, 1]
    frame_idx <- path[p, 2]
    token_frames[[tok_idx]] <- c(token_frames[[tok_idx]], frame_idx)
  }

  # Convert frame indices to timestamps
  # Each audio frame = 2 mel frames (due to conv stride 2)
  # Each mel frame = WHISPER_HOP_LENGTH / WHISPER_SAMPLE_RATE seconds
  seconds_per_frame <- 0.02  # 1500 frames = 30 seconds

  token_starts <- numeric(n_text)
  token_ends <- numeric(n_text)
  for (k in seq_len(n_text)) {
    frames <- token_frames[[k]]
    if (length(frames) > 0) {
      token_starts[k] <- (min(frames) - 1) * seconds_per_frame + time_offset
      token_ends[k] <- max(frames) * seconds_per_frame + time_offset
    } else if (k > 1) {
      # Inherit from previous token
      token_starts[k] <- token_ends[k - 1]
      token_ends[k] <- token_starts[k]
    } else {
      token_starts[k] <- time_offset
      token_ends[k] <- time_offset
    }
  }

  # Group subword tokens into words
  group_into_words(text_token_ids, token_starts, token_ends, tokenizer)
}

#' Group Subword Tokens into Words
#'
#' Merge BPE subword tokens into whole words with timestamps.
#'
#' @param token_ids Integer vector of text token IDs
#' @param starts Numeric vector of token start times
#' @param ends Numeric vector of token end times
#' @param tokenizer Whisper tokenizer
#' @return Data frame with word, start, end columns
group_into_words <- function(
  token_ids,
  starts,
  ends,
  tokenizer
) {
  if (length(token_ids) == 0) {
    return(data.frame(word = character(0), start = numeric(0), end = numeric(0)))
  }

  # Decode each token individually
  token_texts <- vapply(token_ids, function(id) tokenizer$decode(id), character(1))

  # Group by word boundaries (space at start of token = new word)
  words <- list()
  current_word <- ""
  current_start <- starts[1]
  current_end <- ends[1]

  for (i in seq_along(token_texts)) {
    text <- token_texts[i]
    is_new_word <- grepl("^\\s", text) || i == 1L

    if (is_new_word && nchar(trimws(current_word)) > 0 && i > 1L) {
      # Save previous word
      words <- c(words, list(data.frame(
        word = trimws(current_word),
        start = current_start,
        end = current_end,
        stringsAsFactors = FALSE
      )))
      current_word <- text
      current_start <- starts[i]
      current_end <- ends[i]
    } else {
      current_word <- paste0(current_word, text)
      current_end <- ends[i]
    }
  }

  # Save last word
  if (nchar(trimws(current_word)) > 0) {
    words <- c(words, list(data.frame(
      word = trimws(current_word),
      start = current_start,
      end = current_end,
      stringsAsFactors = FALSE
    )))
  }

  if (length(words) == 0) {
    return(data.frame(word = character(0), start = numeric(0), end = numeric(0)))
  }

  do.call(rbind, words)
}

#' DTW Alignment
#'
#' Standard dynamic time warping on a cost matrix.
#'
#' @param cost Numeric matrix (n_tokens x n_frames)
#' @return Integer matrix with 2 columns (token_idx, frame_idx), 1-indexed
dtw_align <- function(cost) {
  n <- nrow(cost)
  m <- ncol(cost)

  # Accumulated cost matrix
  D <- matrix(Inf, nrow = n, ncol = m)
  D[1, 1] <- cost[1, 1]

  # First row: can only come from the left
  if (m >= 2L) {
    for (j in 2:m) {
      D[1, j] <- D[1, j - 1] + cost[1, j]
    }
  }

  # First column: can only come from above
  if (n >= 2L) {
    for (i in 2:n) {
      D[i, 1] <- D[i - 1, 1] + cost[i, 1]
    }
  }

  # Fill rest
  if (n >= 2L && m >= 2L) {
    for (i in 2:n) {
      for (j in 2:m) {
        D[i, j] <- cost[i, j] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
      }
    }
  }

  # Backtrack to find optimal path
  path <- matrix(0L, nrow = n + m, ncol = 2)
  k <- 1L
  i <- n
  j <- m
  path[k, ] <- c(i, j)


  while (i > 1 || j > 1) {
    k <- k + 1L
    if (i == 1) {
      j <- j - 1L
    } else if (j == 1) {
      i <- i - 1L
    } else {
      candidates <- c(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
      step <- which.min(candidates)
      if (step == 1L) {
        i <- i - 1L
        j <- j - 1L
      } else if (step == 2L) {
        i <- i - 1L
      } else {
        j <- j - 1L
      }
    }
    path[k, ] <- c(i, j)
  }

  # Reverse path (was built backwards)
  path <- path[k:1, , drop = FALSE]
  path
}

#' 1D Median Filter
#'
#' Apply a sliding median filter to a numeric vector.
#'
#' @param x Numeric vector
#' @param width Filter width (must be odd)
#' @return Filtered numeric vector of same length
#' @importFrom stats median
medfilt1 <- function(x, width = 7L) {
  n <- length(x)
  if (n == 0) return(x)

  # Ensure odd width
  if (width %% 2L == 0L) width <- width + 1L
  half <- width %/% 2L

  result <- numeric(n)
  for (i in seq_len(n)) {
    lo <- max(1L, i - half)
    hi <- min(n, i + half)
    result[i] <- median(x[lo:hi])
  }
  result
}
