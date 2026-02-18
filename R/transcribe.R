#' Whisper Transcription
#'
#' Main transcription API for Whisper.

#' Create a Whisper Pipeline
#'
#' Load the model, tokenizer, and config once. Call \code{$transcribe()}
#' repeatedly without reloading.
#'
#' @param model Model name: "tiny", "base", "small", "medium", "large-v3"
#' @param device Device: "auto", "cpu", "cuda"
#' @param dtype Data type: "auto", "float16", "float32"
#' @param download If TRUE and model not present, prompt to download.
#' @param verbose Print loading messages.
#' @return A \code{whisper_pipeline} object with a \code{$transcribe()} method.
#' @export
#' @examples
#' \donttest{
#' if (model_exists("tiny")) {
#'   pipe <- whisper_pipeline("tiny")
#'   pipe$transcribe(system.file("audio", "jfk.mp3", package = "whisper"))
#' }
#' }
whisper_pipeline <- function(
  model = "tiny",
  device = "auto",
  dtype = "auto",
  download = TRUE,
  verbose = TRUE
) {
  device <- parse_device(device)
  dtype <- parse_dtype(dtype, device)

  whisper <- load_whisper_model(model, device = device, dtype = dtype,
    download = download, verbose = verbose)
  tokenizer <- whisper_tokenizer(model)
  config <- whisper_config(model)

  # Pre-warm token cache (avoids 200ms hub_download on first transcribe)
  whisper_special_tokens(model)

  pipe <- list(
    model = whisper,
    tokenizer = tokenizer,
    config = config,
    device = device,
    dtype = dtype
  )

  pipe$transcribe <- function(
    file,
    language = "en",
    task = "transcribe",
    timestamps = FALSE,
    word_timestamps = FALSE,
    verbose = TRUE
  ) {
    pipeline_transcribe(pipe, file, language = language, task = task,
      timestamps = timestamps, word_timestamps = word_timestamps,
      verbose = verbose)
  }

  class(pipe) <- "whisper_pipeline"
  pipe
}

#' @export
print.whisper_pipeline <- function(x, ...) {
  cat(sprintf("<whisper_pipeline: %s on %s>\n",
    x$config$model_name, x$device))
  invisible(x)
}

#' Pipeline Transcribe
#'
#' @param pipe A whisper_pipeline object.
#' @param file Path to audio file.
#' @param language Language code.
#' @param task Task type.
#' @param timestamps Return segment-level timestamps.
#' @param word_timestamps Return word-level timestamps.
#' @param verbose Print progress.
#' @return List with text, language, and metadata.
#' @keywords internal
pipeline_transcribe <- function(
  pipe,
  file,
  language = "en",
  task = "transcribe",
  timestamps = FALSE,
  word_timestamps = FALSE,
  verbose = TRUE
) {
  if (!file.exists(file)) stop("Audio file not found: ", file)

  # word_timestamps implies timestamps
  if (word_timestamps) timestamps <- TRUE

  duration <- audio_duration(file)
  if (verbose) message("Audio duration: ", round(duration, 1), "s")

  if (duration <= WHISPER_CHUNK_LENGTH) {
    result <- transcribe_chunk(file, pipe$model, pipe$tokenizer, pipe$config,
      language = language, task = task, timestamps = timestamps,
      word_timestamps = word_timestamps,
      device = pipe$device, dtype = pipe$dtype, verbose = verbose)
  } else {
    result <- transcribe_long(file, pipe$model, pipe$tokenizer, pipe$config,
      language = language, task = task, timestamps = timestamps,
      word_timestamps = word_timestamps,
      device = pipe$device, dtype = pipe$dtype, verbose = verbose)
  }

  result$model <- pipe$config$model_name
  result$backend <- "whisper"
  result$duration <- duration
  result
}

#' Transcribe Audio
#'
#' Transcribe speech from an audio file using Whisper.
#'
#' For repeated transcription, use \code{\link{whisper_pipeline}()} to
#' load the model once.
#'
#' @param file Path to audio file (WAV, MP3, etc.)
#' @param model Model name: "tiny", "base", "small", "medium", "large-v3"
#' @param language Language code (e.g., "en", "es"). NULL for auto-detection.
#' @param task "transcribe" or "translate" (translate to English)
#' @param timestamps If TRUE, return segment-level timestamps
#' @param word_timestamps If TRUE, return word-level timestamps (implies timestamps)
#' @param device Device: "auto", "cpu", "cuda"
#' @param dtype Data type: "auto", "float16", "float32"
#' @param verbose Print progress messages
#' @return List with text, language, and metadata. When \code{timestamps=TRUE},
#'   includes \code{segments} data.frame with start, end, text columns. When
#'   \code{word_timestamps=TRUE}, includes \code{words} data.frame with word,
#'   start, end columns.
#' @export
#' @examples
#' \donttest{
#' # Transcribe included sample (JFK "ask not" speech)
#' if (model_exists("tiny")) {
#'   audio_file <- system.file("audio", "jfk.mp3", package = "whisper")
#'   result <- transcribe(audio_file, model = "tiny")
#'   result$text
#'
#'   # With timestamps
#'   result <- transcribe(audio_file, model = "tiny", timestamps = TRUE)
#'   result$segments
#'
#'   # Translate Spanish audio to English
#'   spanish_file <- system.file("audio", "allende.mp3", package = "whisper")
#'   result <- transcribe(spanish_file, model = "tiny",
#'                        language = "es", task = "translate")
#'   result$text
#' }
#' }
transcribe <- function(
  file,
  model = "tiny",
  language = "en",
  task = "transcribe",
  timestamps = FALSE,
  word_timestamps = FALSE,
  device = "auto",
  dtype = "auto",
  verbose = TRUE
) {
  pipe <- whisper_pipeline(model, device = device, dtype = dtype,
    download = TRUE, verbose = verbose)
  pipe$transcribe(file, language = language, task = task,
    timestamps = timestamps, word_timestamps = word_timestamps,
    verbose = verbose)
}

#' Transcribe Single Chunk
#'
#' @param file Audio file or mel spectrogram
#' @param model WhisperModel
#' @param tokenizer Tokenizer
#' @param config Model config
#' @param language Language code
#' @param task Task type
#' @param device Device
#' @param dtype Dtype
#' @param verbose Verbose output
#' @return Transcription result
transcribe_chunk <- function(
  file,
  model,
  tokenizer,
  config,
  language = "en",
  task = "transcribe",
  timestamps = FALSE,
  word_timestamps = FALSE,
  time_offset = 0,
  device,
  dtype,
  verbose = TRUE
) {
  # Convert audio to mel spectrogram
  if (verbose) message("Processing audio...")
  mel <- audio_to_mel(file, n_mels = config$n_mels, device = device, dtype = dtype)

  # Get initial decoder tokens (use model name for correct special token IDs)
  initial_tokens <- get_initial_tokens(language, task,
    model = config$model_name, timestamps = timestamps)
  tokens <- torch::torch_tensor(matrix(initial_tokens, nrow = 1),
    dtype = torch::torch_long(),
    device = device)

  # Encode audio
  if (verbose) message("Encoding audio...")
  torch::with_no_grad({
      encoder_output <- model$encode(mel)
    })

  # Decode with greedy search
  if (verbose) message("Decoding...")
  decode_result <- greedy_decode(model, encoder_output, tokens, tokenizer,
    max_length = config$n_text_ctx,
    timestamps = timestamps, word_timestamps = word_timestamps,
    device = device)

  # greedy_decode returns list when timestamps/word_timestamps, integer vector otherwise
  if (is.list(decode_result)) {
    generated <- decode_result$tokens
    cross_attn_weights <- decode_result$cross_attn_weights
  } else {
    generated <- decode_result
    cross_attn_weights <- NULL
  }

  # Build result
  if (timestamps) {
    # Extract segments from timestamp tokens
    segments <- extract_segments(generated, tokenizer, time_offset = time_offset)
    text <- paste(segments$text, collapse = " ")
    text <- clean_text(text)
    result <- list(text = text, language = language, segments = segments)
  } else {
    text <- tokenizer$decode(generated)
    text <- clean_text(text)
    result <- list(text = text, language = language)
  }

  # Word-level timestamps via cross-attention DTW
  if (word_timestamps && !is.null(cross_attn_weights)) {
    special <- whisper_special_tokens(config$model_name)
    sample_begin <- length(initial_tokens)
    words <- compute_word_timestamps(generated, cross_attn_weights,
      tokenizer, config, time_offset = time_offset,
      sample_begin = sample_begin)
    result$words <- words
  }

  result
}

#' Greedy Decoding
#'
#' @param model WhisperModel
#' @param encoder_output Encoder hidden states
#' @param initial_tokens Initial token tensor
#' @param tokenizer Tokenizer
#' @param max_length Maximum output length
#' @param timestamps Whether to allow timestamp tokens
#' @param word_timestamps Whether to collect cross-attention weights
#' @param device Device
#' @return Integer vector of generated tokens, or list with tokens and
#'   cross_attn_weights when word_timestamps is TRUE
greedy_decode <- function(
  model,
  encoder_output,
  initial_tokens,
  tokenizer,
  max_length = 448L,
  timestamps = FALSE,
  word_timestamps = FALSE,
  device
) {
  # Use model-specific special tokens
  special <- whisper_special_tokens(tokenizer$model)
  generated <- as.integer(as.array(initial_tokens$cpu()))
  sample_begin <- length(generated)

  kv_cache <- NULL
  tokens <- initial_tokens
  need_weights <- word_timestamps

  # Collect cross-attention weights for word timestamps
  all_cross_attn <- if (word_timestamps) list() else NULL

  torch::with_no_grad({
      for (i in seq_len(max_length)) {
        # Stop if we've reached max context length
        if (length(generated) >= max_length) break

        # Get next token logits
        result <- model$decode(tokens, encoder_output, kv_cache = kv_cache,
          need_weights = need_weights)
        logits <- result$logits
        kv_cache <- result$kv_cache

        # Get last position logits (R uses 1-based indexing)
        seq_len_val <- logits$size(2)
        next_logits <- logits[, seq_len_val,]# (batch, vocab)

        # Apply timestamp logit rules when timestamps are enabled
        if (timestamps) {
          next_logits <- apply_timestamp_rules(next_logits, generated,
            special, sample_begin)
        }

        # Greedy: take argmax (subtract 1 because R torch argmax returns 1-indexed)
        next_token <- next_logits$argmax(dim = - 1L)
        next_token_id <- as.integer(next_token$item()) - 1L

        # Check for end of text
        if (next_token_id == special$eot) {
          break
        }

        # Append token
        generated <- c(generated, next_token_id)

        # Collect cross-attention weights for this step
        if (word_timestamps && !is.null(result$cross_attn_weights)) {
          all_cross_attn <- c(all_cross_attn, list(result$cross_attn_weights))
        }

        # Prepare next input (decoder expects 0-indexed token IDs, adds 1 internally)
        tokens <- torch::torch_tensor(matrix(next_token_id, nrow = 1L),
          dtype = torch::torch_long(),
          device = device)
      }
    })

  if (word_timestamps) {
    list(tokens = generated, cross_attn_weights = all_cross_attn)
  } else if (timestamps) {
    list(tokens = generated, cross_attn_weights = NULL)
  } else {
    generated
  }
}

#' Apply Timestamp Token Rules
#'
#' Enforce Whisper timestamp generation constraints on logits.
#'
#' @param logits Logit tensor (1, vocab) or (vocab)
#' @param generated Integer vector of tokens generated so far
#' @param special Special token IDs
#' @param sample_begin Index where content tokens start in generated
#' @return Modified logits tensor
apply_timestamp_rules <- function(
  logits,
  generated,
  special,
  sample_begin
) {
  # Content tokens are those generated after the initial prompt tokens
  content_tokens <- generated[seq_len(length(generated)) > sample_begin]
  ts_begin <- special$timestamp_begin
  # Max timestamp: 30.00s = 1500 steps of 0.02s

  max_ts <- ts_begin + 1500L

  # Determine if logits are 1D (vocab) or 2D (batch, vocab)
  is_2d <- logits$dim() == 2L

  # Rule 1: First content token must be a timestamp (<|0.00|>)
  if (length(content_tokens) == 0) {
    # Suppress all non-timestamp tokens
    if (is_2d) {
      logits[, 1:ts_begin] <- -Inf
    } else {
      logits[1:ts_begin] <- -Inf
    }
    # Only allow <|0.00|> (first timestamp)
    if (max_ts > ts_begin + 1L) {
      if (is_2d) {
        logits[, (ts_begin + 2L):logits$size(2)] <- -Inf
      } else {
        logits[(ts_begin + 2L):logits$size(1)] <- -Inf
      }
    }
    return(logits)
  }

  # Find last timestamp in content tokens
  last_ts <- NA
  for (j in rev(seq_along(content_tokens))) {
    if (content_tokens[j] >= ts_begin) {
      last_ts <- content_tokens[j]
      break
    }
  }

  # Count consecutive timestamps at end
  n_consecutive_ts <- 0L
  for (j in rev(seq_along(content_tokens))) {
    if (content_tokens[j] >= ts_begin) {
      n_consecutive_ts <- n_consecutive_ts + 1L
    } else {
      break
    }
  }

  # Rule 2: After a closing timestamp (2 consecutive), next must be timestamp or EOT
  if (n_consecutive_ts >= 2L && n_consecutive_ts %% 2L == 0L) {
    # Suppress all text tokens, allow only timestamps and EOT
    if (is_2d) {
      # Suppress everything except EOT and timestamps
      mask <- rep(-Inf, logits$size(2))
      mask[special$eot + 1L] <- 0  # Allow EOT (1-indexed)
      mask[(ts_begin + 1L):length(mask)] <- 0  # Allow timestamps
      logits <- logits + torch::torch_tensor(matrix(mask, nrow = 1),
        device = logits$device, dtype = logits$dtype)
    } else {
      mask <- rep(-Inf, logits$size(1))
      mask[special$eot + 1L] <- 0
      mask[(ts_begin + 1L):length(mask)] <- 0
      logits <- logits + torch::torch_tensor(mask,
        device = logits$device, dtype = logits$dtype)
    }
  }

  # Rule 3: After a single timestamp (odd count), next must be non-timestamp (text)
  if (n_consecutive_ts >= 1L && n_consecutive_ts %% 2L == 1L) {
    # Suppress timestamps
    n_vocab <- if (is_2d) logits$size(2) else logits$size(1)
    if (n_vocab > ts_begin) {
      if (is_2d) {
        logits[, (ts_begin + 1L):n_vocab] <- -Inf
      } else {
        logits[(ts_begin + 1L):n_vocab] <- -Inf
      }
    }
  }

  # Rule 4: No backwards timestamps (suppress tokens below last emitted timestamp)
  if (!is.na(last_ts) && last_ts >= ts_begin) {
    suppress_up_to <- last_ts  # Suppress all timestamps <= last_ts
    if (suppress_up_to >= ts_begin) {
      if (is_2d) {
        logits[, (ts_begin + 1L):(suppress_up_to + 1L)] <- -Inf
      } else {
        logits[(ts_begin + 1L):(suppress_up_to + 1L)] <- -Inf
      }
    }
  }

  # Rule 5: Cap max timestamp at 30.00s
  n_vocab <- if (is_2d) logits$size(2) else logits$size(1)
  if (n_vocab > max_ts + 1L) {
    if (is_2d) {
      logits[, (max_ts + 2L):n_vocab] <- -Inf
    } else {
      logits[(max_ts + 2L):n_vocab] <- -Inf
    }
  }

  logits
}

#' Transcribe Long Audio
#'
#' Process audio longer than 30 seconds in chunks.
#'
#' @param file Audio file
#' @param model WhisperModel
#' @param tokenizer Tokenizer
#' @param config Model config
#' @param language Language
#' @param task Task
#' @param device Device
#' @param dtype Dtype
#' @param verbose Verbose
#' @return Combined transcription result
transcribe_long <- function(
  file,
  model,
  tokenizer,
  config,
  language,
  task,
  timestamps = FALSE,
  word_timestamps = FALSE,
  device,
  dtype,
  verbose
) {
  # Split into chunks
  chunk_length <- 30
  overlap <- 1
  hop_seconds <- chunk_length - overlap
  chunks <- split_audio(file, chunk_length = chunk_length, overlap = overlap)

  if (verbose) message("Processing ", length(chunks), " chunks...")

  all_text <- character(length(chunks))
  all_segments <- if (timestamps) list() else NULL
  all_words <- if (word_timestamps) list() else NULL

  for (i in seq_along(chunks)) {
    if (verbose) message("  Chunk ", i, "/", length(chunks))
    time_offset <- (i - 1) * hop_seconds

    # Transcribe chunk with time offset
    chunk_result <- transcribe_chunk(chunks[[i]], model, tokenizer, config,
      language = language, task = task, timestamps = timestamps,
      word_timestamps = word_timestamps, time_offset = time_offset,
      device = device, dtype = dtype, verbose = FALSE)

    all_text[i] <- chunk_result$text

    if (timestamps && !is.null(chunk_result$segments) && nrow(chunk_result$segments) > 0) {
      all_segments <- c(all_segments, list(chunk_result$segments))
    }

    if (word_timestamps && !is.null(chunk_result$words) && nrow(chunk_result$words) > 0) {
      all_words <- c(all_words, list(chunk_result$words))
    }
  }

  # Combine results
  result <- list(
    text = paste(all_text, collapse = " "),
    language = language
  )

  if (timestamps) {
    result$segments <- if (length(all_segments) > 0) {
      do.call(rbind, all_segments)
    } else {
      data.frame(start = numeric(0), end = numeric(0), text = character(0))
    }
  }

  if (word_timestamps) {
    result$words <- if (length(all_words) > 0) {
      do.call(rbind, all_words)
    } else {
      data.frame(word = character(0), start = numeric(0), end = numeric(0))
    }
  }

  result
}

#' Clean Transcribed Text
#'
#' @param text Raw decoded text
#' @return Cleaned text
clean_text <- function(text) {
  # Remove special tokens that might have leaked through
  text <- gsub("<\\|[^>]+\\|>", "", text)

  # Trim whitespace

  text <- trimws(text)

  # Collapse multiple spaces
  text <- gsub("\\s+", " ", text)

  text
}

#' Extract Segments with Timestamps
#'
#' @param tokens Token IDs
#' @param tokenizer Tokenizer
#' @param time_offset Offset in seconds for chunk processing
#' @return Data frame with start, end, text
extract_segments <- function(
  tokens,
  tokenizer,
  time_offset = 0
) {
  # Use model-specific special tokens
  model_name <- tokenizer$model
  special <- whisper_special_tokens(model_name)

  segments <- list()
  current_start <- time_offset
  current_tokens <- integer(0)

  for (tok in tokens) {
    if (is_timestamp_token(tok, model_name)) {
      timestamp <- decode_timestamp(tok, model_name) + time_offset

      if (length(current_tokens) > 0) {
        # End of segment
        text <- tokenizer$decode(current_tokens)
        text <- clean_text(text)

        if (nchar(text) > 0) {
          segments <- c(segments, list(data.frame(
                start = current_start,
                end = timestamp,
                text = text,
                stringsAsFactors = FALSE
              )))
        }

        current_tokens <- integer(0)
      }

      current_start <- timestamp
    } else if (tok >= special$sot && tok < special$timestamp_begin) {
      # Skip special tokens
      next
    } else {
      current_tokens <- c(current_tokens, tok)
    }
  }

  # Handle remaining tokens
  if (length(current_tokens) > 0) {
    text <- tokenizer$decode(current_tokens)
    text <- clean_text(text)

    if (nchar(text) > 0) {
      segments <- c(segments, list(data.frame(
            start = current_start,
            end = current_start + 0.5, # Estimate
            text = text,
            stringsAsFactors = FALSE
          )))
    }
  }

  if (length(segments) == 0) {
    return(data.frame(start = numeric(0), end = numeric(0), text = character(0)))
  }

  do.call(rbind, segments)
}

