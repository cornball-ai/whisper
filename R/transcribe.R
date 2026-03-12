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

  pipe <- list(
    model = whisper,
    tokenizer = tokenizer,
    config = config,
    device = device,
    dtype = dtype
  )

  pipe$transcribe <- function(
    file,
    language = NULL,
    task = "transcribe",
    timestamps = FALSE,
    word_timestamps = FALSE,
    beam_size = 1L,
    temperatures = 0,
    best_of = 1L,
    compression_ratio_threshold = 2.4,
    logprob_threshold = -1.0,
    length_penalty = 1.0,
    patience = Inf,
    verbose = TRUE
  ) {
    pipeline_transcribe(pipe, file, language = language, task = task,
      timestamps = timestamps, word_timestamps = word_timestamps,
      beam_size = beam_size, temperatures = temperatures,
      best_of = best_of,
      compression_ratio_threshold = compression_ratio_threshold,
      logprob_threshold = logprob_threshold,
      length_penalty = length_penalty, patience = patience,
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
#' @param beam_size Number of beams for beam search.
#' @param temperatures Numeric vector of temperatures for fallback.
#' @param best_of Number of samples per temperature > 0.
#' @param compression_ratio_threshold Max compression ratio before fallback.
#' @param logprob_threshold Min average log probability before fallback.
#' @param length_penalty Length penalty exponent for beam search.
#' @param patience Patience factor for beam search.
#' @param verbose Print progress.
#' @return List with text, language, and metadata.
#' @keywords internal
pipeline_transcribe <- function(
  pipe,
  file,
  language = NULL,
  task = "transcribe",
  timestamps = FALSE,
  word_timestamps = FALSE,
  beam_size = 1L,
  temperatures = 0,
  best_of = 1L,
  compression_ratio_threshold = 2.4,
  logprob_threshold = -1.0,
  length_penalty = 1.0,
  patience = Inf,
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
      beam_size = beam_size, temperatures = temperatures,
      best_of = best_of,
      compression_ratio_threshold = compression_ratio_threshold,
      logprob_threshold = logprob_threshold,
      length_penalty = length_penalty, patience = patience,
      device = pipe$device, dtype = pipe$dtype, verbose = verbose)
  } else {
    result <- transcribe_long(file, pipe$model, pipe$tokenizer, pipe$config,
      language = language, task = task, timestamps = timestamps,
      word_timestamps = word_timestamps,
      beam_size = beam_size, temperatures = temperatures,
      best_of = best_of,
      compression_ratio_threshold = compression_ratio_threshold,
      logprob_threshold = logprob_threshold,
      length_penalty = length_penalty, patience = patience,
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
#' @param language Language code (e.g., "en", "es"), or NULL (default) for
#'   auto-detection from the audio.
#' @param task "transcribe" or "translate" (translate to English)
#' @param timestamps If TRUE, return segment-level timestamps
#' @param word_timestamps If TRUE, return word-level timestamps (implies timestamps)
#' @param beam_size Number of beams for beam search (1 = greedy, default)
#' @param temperatures Numeric vector of temperatures to try. 0 uses beam search
#'   or greedy; values > 0 use sampling. Multiple values enable fallback.
#' @param best_of Number of samples per temperature > 0, keeping the best.
#' @param compression_ratio_threshold Max compression ratio before fallback.
#' @param logprob_threshold Min average log probability before fallback.
#' @param length_penalty Length penalty exponent for beam search scoring.
#' @param patience Patience factor for beam search (stop after patience*beam_size).
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
#' if (model_exists("tiny")) {
#'   audio_file <- system.file("audio", "jfk.mp3", package = "whisper")
#'
#'   # Auto-detect language (default)
#'   result <- transcribe(audio_file, model = "tiny")
#'   result$language  # "en"
#'   result$text
#'
#'   # Explicit language
#'   result <- transcribe(audio_file, model = "tiny", language = "en")
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
  language = NULL,
  task = "transcribe",
  timestamps = FALSE,
  word_timestamps = FALSE,
  beam_size = 1L,
  temperatures = 0,
  best_of = 1L,
  compression_ratio_threshold = 2.4,
  logprob_threshold = -1.0,
  length_penalty = 1.0,
  patience = Inf,
  device = "auto",
  dtype = "auto",
  verbose = TRUE
) {
  pipe <- whisper_pipeline(model, device = device, dtype = dtype,
    download = TRUE, verbose = verbose)
  pipe$transcribe(file, language = language, task = task,
    timestamps = timestamps, word_timestamps = word_timestamps,
    beam_size = beam_size, temperatures = temperatures,
    best_of = best_of,
    compression_ratio_threshold = compression_ratio_threshold,
    logprob_threshold = logprob_threshold,
    length_penalty = length_penalty, patience = patience,
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
#' @param timestamps Return segment-level timestamps.
#' @param word_timestamps Return word-level timestamps.
#' @param beam_size Number of beams for beam search.
#' @param temperatures Numeric vector of temperatures for fallback.
#' @param best_of Number of samples per temperature > 0.
#' @param compression_ratio_threshold Max compression ratio before fallback.
#' @param logprob_threshold Min average log probability before fallback.
#' @param length_penalty Length penalty exponent for beam search.
#' @param patience Patience factor for beam search.
#' @param time_offset Time offset in seconds for chunk processing.
#' @param device Device
#' @param dtype Dtype
#' @param verbose Verbose output
#' @return Transcription result
transcribe_chunk <- function(
  file,
  model,
  tokenizer,
  config,
  language = NULL,
  task = "transcribe",
  timestamps = FALSE,
  word_timestamps = FALSE,
  beam_size = 1L,
  temperatures = 0,
  best_of = 1L,
  compression_ratio_threshold = 2.4,
  logprob_threshold = -1.0,
  length_penalty = 1.0,
  patience = Inf,
  time_offset = 0,
  device,
  dtype,
  verbose = TRUE
) {
  # Convert audio to mel spectrogram (full 30s window)
  if (verbose) message("Processing audio...")
  full_mel <- audio_to_mel(file, n_mels = config$n_mels, device = device, dtype = dtype)
  n_frames <- full_mel$size(3)  # 3000 for 30s

  # Beam search needs timestamps internally for proper termination
  user_timestamps <- timestamps
  internal_timestamps <- timestamps || beam_size > 1L

  special <- whisper_special_tokens(config$model_name)

  # Auto-detect language if not specified
  if (is.null(language)) {
    detection <- detect_language_from_mel(model, full_mel, config, device)
    language <- detection$language
    if (verbose) {
      top <- detection$probabilities[1]
      message("Detected language: ", language,
        " (", round(top * 100, 1), "%)")
    }
  }

  # Seek loop: decode repeatedly, advancing through the mel spectrogram
  seek <- 0L  # current frame position
  all_generated <- integer(0)
  all_cross_attn <- if (word_timestamps) list() else NULL
  all_segments <- list()
  seek_iter <- 0L

  while (seek < n_frames) {
    seek_iter <- seek_iter + 1L
    if (seek_iter > 50L) break  # safety limit

    # Slice mel from seek position, pad to full width
    remaining <- n_frames - seek
    if (remaining < 1L) break

    if (seek == 0L) {
      mel <- full_mel
    } else if (seek + 1L > n_frames) {
      break
    } else {
      # Slice mel[1, :, seek:] and pad to n_frames width
      mel_slice <- full_mel[, , (seek + 1L):n_frames]
      pad_width <- n_frames - mel_slice$size(3)
      if (pad_width > 0L) {
        mel <- torch::nnf_pad(mel_slice, c(0L, pad_width), value = 0)
      } else {
        mel <- mel_slice
      }
    }

    # Compute seek time for this iteration
    seek_time <- seek * 0.01  # frames to seconds (10ms per frame)

    # Get initial decoder tokens
    initial_tokens <- get_initial_tokens(language, task,
      model = config$model_name, timestamps = internal_timestamps)
    tokens <- torch::torch_tensor(matrix(initial_tokens, nrow = 1),
      dtype = torch::torch_long(), device = device)

    # Encode audio
    torch::with_no_grad({
      encoder_output <- model$encode(mel)
    })

    # Decode
    decode_result <- decode_with_fallback(model, encoder_output, tokens,
      tokenizer, temperatures = temperatures, beam_size = beam_size,
      best_of = best_of, max_length = config$n_text_ctx,
      timestamps = internal_timestamps, word_timestamps = word_timestamps,
      compression_ratio_threshold = compression_ratio_threshold,
      logprob_threshold = logprob_threshold,
      length_penalty = length_penalty, patience = patience,
      device = device)

    generated <- decode_result$tokens
    all_generated <- c(all_generated, generated)

    # Find the last timestamp token to determine where to seek next
    last_ts_frame <- 0L
    for (tok in generated) {
      if (tok >= special$timestamp_begin) {
        ts_seconds <- (tok - special$timestamp_begin) * 0.02
        ts_frame <- as.integer(ts_seconds * 100)  # seconds to frames (10ms)
        if (ts_frame > last_ts_frame) last_ts_frame <- ts_frame
      }
    }

    # Extract segments with proper time offset
    if (user_timestamps) {
      segments <- extract_segments(generated, tokenizer,
        time_offset = time_offset + seek_time)
      if (nrow(segments) > 0) {
        all_segments <- c(all_segments, list(segments))
      }
    }

    # Collect cross-attention weights with seek offset
    if (word_timestamps && !is.null(decode_result$cross_attn_weights)) {
      all_cross_attn <- c(all_cross_attn, list(list(
        weights = decode_result$cross_attn_weights,
        tokens = generated,
        initial_tokens = initial_tokens,
        seek_time = seek_time
      )))
    }

    # Advance seek position
    if (last_ts_frame > 0L) {
      seek <- seek + last_ts_frame
    } else {
      # No timestamp found — model produced no timed output, skip ahead
      break
    }

    # If last timestamp covered nearly the full remaining audio, stop
    if (last_ts_frame >= remaining - 100L) break
  }

  if (verbose && seek_iter > 1L) {
    message("  Seek loop: ", seek_iter, " iterations")
  }

  # Build result
  if (user_timestamps) {
    if (length(all_segments) > 0) {
      segments <- do.call(rbind, all_segments)
      text <- paste(segments$text, collapse = " ")
      text <- clean_text(text)
    } else {
      segments <- data.frame(start = numeric(0), end = numeric(0),
        text = character(0))
      text <- ""
    }
    result <- list(text = text, language = language, segments = segments)
  } else {
    # For non-timestamp mode, combine all generated tokens
    # (strip timestamp tokens if used internally)
    if (internal_timestamps) {
      all_generated <- all_generated[all_generated < special$timestamp_begin]
    }
    text <- tokenizer$decode(all_generated)
    text <- clean_text(text)
    result <- list(text = text, language = language)
  }

  # Word-level timestamps via cross-attention DTW (per seek iteration)
  if (word_timestamps && length(all_cross_attn) > 0) {
    all_words <- list()
    for (ca in all_cross_attn) {
      sample_begin <- length(ca$initial_tokens)
      words <- compute_word_timestamps(ca$tokens, ca$weights,
        tokenizer, config,
        time_offset = time_offset + ca$seek_time,
        sample_begin = sample_begin)
      if (!is.null(words) && nrow(words) > 0) {
        all_words <- c(all_words, list(words))
      }
    }
    if (length(all_words) > 0) {
      result$words <- do.call(rbind, all_words)
    }
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
  sum_logprob <- 0
  n_tokens <- 0L

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
        next_logits <- logits[, seq_len_val, ] # (batch, vocab)

        # Apply timestamp logit rules when timestamps are enabled
        if (timestamps) {
          next_logits <- apply_timestamp_rules(next_logits, generated,
            special, sample_begin)
        }

        # Accumulate log probability before argmax
        log_probs <- torch::nnf_log_softmax(next_logits, dim = -1L)

        # Greedy: take argmax (subtract 1 because R torch argmax returns 1-indexed)
        next_token <- next_logits$argmax(dim = -1L)
        next_token_id <- as.integer(next_token$item()) - 1L

        # Track log prob of selected token (1-indexed for tensor access)
        sum_logprob <- sum_logprob + as.numeric(log_probs[1, next_token$item()]$item())
        n_tokens <- n_tokens + 1L

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

  list(
    tokens = generated,
    cross_attn_weights = all_cross_attn,
    sum_logprob = sum_logprob,
    n_tokens = n_tokens
  )
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
#' @param timestamps Return segment-level timestamps.
#' @param word_timestamps Return word-level timestamps.
#' @param beam_size Number of beams for beam search.
#' @param temperatures Numeric vector of temperatures for fallback.
#' @param best_of Number of samples per temperature > 0.
#' @param compression_ratio_threshold Max compression ratio before fallback.
#' @param logprob_threshold Min average log probability before fallback.
#' @param length_penalty Length penalty exponent for beam search.
#' @param patience Patience factor for beam search.
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
  beam_size = 1L,
  temperatures = 0,
  best_of = 1L,
  compression_ratio_threshold = 2.4,
  logprob_threshold = -1.0,
  length_penalty = 1.0,
  patience = Inf,
  device,
  dtype,
  verbose
) {
  # Auto-detect language from first 30s if not specified
  if (is.null(language)) {
    mel <- audio_to_mel(file, n_mels = config$n_mels, device = device,
      dtype = dtype)
    detection <- detect_language_from_mel(model, mel, config, device)
    language <- detection$language
    if (verbose) {
      top <- detection$probabilities[1]
      message("Detected language: ", language,
        " (", round(top * 100, 1), "%)")
    }
  }

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
      word_timestamps = word_timestamps,
      beam_size = beam_size, temperatures = temperatures,
      best_of = best_of,
      compression_ratio_threshold = compression_ratio_threshold,
      logprob_threshold = logprob_threshold,
      length_penalty = length_penalty, patience = patience,
      time_offset = time_offset,
      device = device, dtype = dtype, verbose = FALSE)

    all_text[i] <- chunk_result$text

    if (timestamps && !is.null(chunk_result$segments) && nrow(chunk_result$segments) > 0) {
      all_segments <- c(all_segments, list(chunk_result$segments))
    }

    if (word_timestamps && !is.null(chunk_result$words) && nrow(chunk_result$words) > 0) {
      all_words <- c(all_words, list(chunk_result$words))
    }
  }

  # Get actual audio duration to filter hallucinations from padded chunks
  audio_dur <- audio_duration(file)

  # Combine results
  result <- list(
    text = paste(all_text, collapse = " "),
    language = language
  )

  if (timestamps) {
    if (length(all_segments) > 0) {
      combined <- do.call(rbind, all_segments)
      # Remove segments that start after the actual audio duration
      combined <- combined[combined$start < audio_dur, , drop = FALSE]
      # Cap end times to audio duration
      combined$end <- pmin(combined$end, audio_dur)
      # Deduplicate overlapping segments at chunk boundaries.
      # Strategy: when two segments overlap, keep the later one (from the
      # chunk that has actual audio for that time region) unless it looks
      # hallucinated (very short text).
      if (nrow(combined) > 1) {
        keep <- rep(TRUE, nrow(combined))
        for (j in 2:nrow(combined)) {
          if (combined$start[j] < combined$end[j - 1] - 0.1) {
            # Overlap detected
            prev_len <- nchar(combined$text[j - 1])
            curr_len <- nchar(combined$text[j])
            if (curr_len < 5) {
              # Later segment is likely hallucination, drop it
              keep[j] <- FALSE
            } else if (prev_len < 5) {
              # Previous segment is likely hallucination, drop it
              keep[j - 1] <- FALSE
            } else {
              # Both substantial: trim previous to end before overlap starts
              combined$end[j - 1] <- combined$start[j]
            }
          }
        }
        combined <- combined[keep, , drop = FALSE]
      }
      # Remove likely hallucinated segments (very short duration + short text)
      seg_dur <- combined$end - combined$start
      combined <- combined[!(seg_dur < 0.5 & nchar(combined$text) < 15), , drop = FALSE]
      result$segments <- combined
    } else {
      result$segments <- data.frame(start = numeric(0), end = numeric(0),
        text = character(0))
    }
  }

  if (word_timestamps) {
    if (length(all_words) > 0) {
      combined_words <- do.call(rbind, all_words)
      # Remove words that start after actual audio duration
      combined_words <- combined_words[combined_words$start < audio_dur, , drop = FALSE]
      # Cap word end times
      combined_words$end <- pmin(combined_words$end, audio_dur)
      # Remove duplicate words from chunk overlap (keep first occurrence)
      if (nrow(combined_words) > 1) {
        keep <- rep(TRUE, nrow(combined_words))
        for (j in 2:nrow(combined_words)) {
          if (combined_words$start[j] < combined_words$end[j - 1] - 0.05) {
            keep[j] <- FALSE
          }
        }
        combined_words <- combined_words[keep, , drop = FALSE]
      }
      # Filter words to only those within retained segments
      if (!is.null(result$segments) && nrow(result$segments) > 0) {
        seg_end <- max(result$segments$end)
        combined_words <- combined_words[combined_words$start < seg_end, , drop = FALSE]
      }
      result$words <- combined_words
    } else {
      result$words <- data.frame(word = character(0), start = numeric(0),
        end = numeric(0))
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

#' Compression Ratio
#'
#' Ratio of raw to compressed text size. High values indicate repetitive
#' or hallucinated output.
#'
#' @param text Character string
#' @return Numeric compression ratio
compression_ratio <- function(text) {
  raw_bytes <- charToRaw(text)
  compressed <- memCompress(raw_bytes, "gzip")
  length(raw_bytes) / length(compressed)
}

#' Rearrange KV Cache by Beam Indices
#'
#' Reorder cached key-value tensors to match new beam ordering.
#'
#' @param kv_cache List of per-layer KV caches
#' @param beam_indices Integer tensor of beam indices (1-indexed)
#' @param device Device
#' @return Reordered KV cache
rearrange_kv_cache <- function(kv_cache, beam_indices, device) {
  lapply(kv_cache, function(layer) {
    reorder_kv <- function(kv) {
      if (is.null(kv)) return(NULL)
      list(
        k = if (!is.null(kv$k)) kv$k$index_select(1L, beam_indices) else NULL,
        v = if (!is.null(kv$v)) kv$v$index_select(1L, beam_indices) else NULL
      )
    }
    list(
      self = reorder_kv(layer$self),
      cross = reorder_kv(layer$cross)
    )
  })
}

#' Expand KV Cache for Beam Search
#'
#' Replicate batch=1 KV cache to batch=beam_size.
#'
#' @param kv_cache List of per-layer KV caches (batch=1)
#' @param beam_size Number of beams
#' @return Expanded KV cache (batch=beam_size)
expand_kv_cache <- function(kv_cache, beam_size) {
  lapply(kv_cache, function(layer) {
    expand_kv <- function(kv) {
      if (is.null(kv)) return(NULL)
      list(
        k = if (!is.null(kv$k)) kv$k$`repeat`(c(beam_size, 1L, 1L, 1L)) else NULL,
        v = if (!is.null(kv$v)) kv$v$`repeat`(c(beam_size, 1L, 1L, 1L)) else NULL
      )
    }
    list(
      self = expand_kv(layer$self),
      cross = expand_kv(layer$cross)
    )
  })
}

#' Sample Decode
#'
#' Temperature-scaled sampling decode. Fork of greedy_decode that
#' uses categorical sampling instead of argmax.
#'
#' @param model WhisperModel
#' @param encoder_output Encoder hidden states
#' @param initial_tokens Initial token tensor (batch=1)
#' @param tokenizer Tokenizer
#' @param temperature Sampling temperature (must be > 0)
#' @param max_length Maximum output length
#' @param timestamps Whether to allow timestamp tokens
#' @param word_timestamps Whether to collect cross-attention weights
#' @param device Device
#' @return List with tokens, cross_attn_weights, sum_logprob, n_tokens
sample_decode <- function(
  model,
  encoder_output,
  initial_tokens,
  tokenizer,
  temperature = 0.6,
  max_length = 448L,
  timestamps = FALSE,
  word_timestamps = FALSE,
  device
) {
  special <- whisper_special_tokens(tokenizer$model)
  generated <- as.integer(as.array(initial_tokens$cpu()))
  sample_begin <- length(generated)

  kv_cache <- NULL
  tokens <- initial_tokens
  need_weights <- word_timestamps

  all_cross_attn <- if (word_timestamps) list() else NULL
  sum_logprob <- 0
  n_tokens <- 0L

  torch::with_no_grad({
    for (i in seq_len(max_length)) {
      if (length(generated) >= max_length) break

      result <- model$decode(tokens, encoder_output, kv_cache = kv_cache,
        need_weights = need_weights)
      logits <- result$logits
      kv_cache <- result$kv_cache

      seq_len_val <- logits$size(2)
      next_logits <- logits[, seq_len_val, ]

      if (timestamps) {
        next_logits <- apply_timestamp_rules(next_logits, generated,
          special, sample_begin)
      }

      # Temperature-scaled sampling
      scaled_logits <- next_logits / temperature
      log_probs <- torch::nnf_log_softmax(scaled_logits, dim = -1L)
      probs <- torch::nnf_softmax(scaled_logits, dim = -1L)
      next_token <- torch::torch_multinomial(probs$squeeze(1L), num_samples = 1L)
      next_token_id <- as.integer(next_token$item()) - 1L

      # Accumulate log probability
      sum_logprob <- sum_logprob + as.numeric(log_probs[1, next_token$item()]$item())
      n_tokens <- n_tokens + 1L

      if (next_token_id == special$eot) break

      generated <- c(generated, next_token_id)

      if (word_timestamps && !is.null(result$cross_attn_weights)) {
        all_cross_attn <- c(all_cross_attn, list(result$cross_attn_weights))
      }

      tokens <- torch::torch_tensor(matrix(next_token_id, nrow = 1L),
        dtype = torch::torch_long(), device = device)
    }
  })

  list(
    tokens = generated,
    cross_attn_weights = all_cross_attn,
    sum_logprob = sum_logprob,
    n_tokens = n_tokens
  )
}

#' Forced Decode
#'
#' Teacher-forcing decode: feed known token sequence one at a time,
#' collecting cross-attention weights. Used by beam search when
#' word_timestamps is needed.
#'
#' @param model WhisperModel
#' @param encoder_output Encoder hidden states
#' @param token_ids Integer vector of all token IDs (including initial)
#' @param device Device
#' @return List of cross-attention weight lists (one per content step)
forced_decode <- function(
  model,
  encoder_output,
  token_ids,
  device
) {
  all_cross_attn <- list()
  kv_cache <- NULL

  # Feed initial tokens as a batch first
  initial <- torch::torch_tensor(matrix(token_ids, nrow = 1L),
    dtype = torch::torch_long(), device = device)

  torch::with_no_grad({
    # Process all tokens, collecting weights at each step after the first
    # We need to process one token at a time to get per-step weights
    for (i in seq_along(token_ids)) {
      tok <- torch::torch_tensor(
        matrix(token_ids[i], nrow = 1L),
        dtype = torch::torch_long(), device = device)

      result <- model$decode(tok, encoder_output, kv_cache = kv_cache,
        need_weights = TRUE)
      kv_cache <- result$kv_cache

      if (i > 1 && !is.null(result$cross_attn_weights)) {
        all_cross_attn <- c(all_cross_attn, list(result$cross_attn_weights))
      }
    }
  })

  all_cross_attn
}

#' Beam Search Decode
#'
#' Beam search decoding for Whisper. Maintains multiple hypotheses
#' and selects the best one based on length-normalized log probability.
#'
#' @param model WhisperModel
#' @param encoder_output Encoder hidden states (batch=1)
#' @param initial_tokens Initial token tensor (batch=1)
#' @param tokenizer Tokenizer
#' @param beam_size Number of beams
#' @param max_length Maximum output length
#' @param timestamps Whether to allow timestamp tokens
#' @param word_timestamps Whether to collect cross-attention weights
#' @param length_penalty Length penalty exponent
#' @param patience Patience factor (stop after patience*beam_size finished)
#' @param device Device
#' @return List with tokens, cross_attn_weights, sum_logprob, n_tokens
beam_search_decode <- function(
  model,
  encoder_output,
  initial_tokens,
  tokenizer,
  beam_size = 5L,
  max_length = 448L,
  timestamps = FALSE,
  word_timestamps = FALSE,
  length_penalty = 1.0,
  patience = Inf,
  device
) {
  special <- whisper_special_tokens(tokenizer$model)
  init_ids <- as.integer(as.array(initial_tokens$cpu()))
  sample_begin <- length(init_ids)
  max_candidates <- if (is.finite(patience)) {
    as.integer(ceiling(patience * beam_size))
  } else {
    .Machine$integer.max
  }

  # Step 1: Run initial prompt tokens with batch=1, get KV cache
  torch::with_no_grad({
    result <- model$decode(initial_tokens, encoder_output, kv_cache = NULL,
      need_weights = FALSE)
  })
  kv_cache <- result$kv_cache
  logits <- result$logits

  # Get first token logits
  seq_len_val <- logits$size(2)
  first_logits <- logits[, seq_len_val, ]

  if (timestamps) {
    first_logits <- apply_timestamp_rules(first_logits, init_ids,
      special, sample_begin)
  }

  first_log_probs <- torch::nnf_log_softmax(first_logits, dim = -1L)

  # Get top beam_size tokens for initial beams
  top <- torch::torch_topk(first_log_probs$squeeze(1L), beam_size)
  top_log_probs <- as.numeric(as.array(top[[1]]$cpu()))
  top_token_ids <- as.integer(as.array(top[[2]]$cpu())) - 1L  # 1-indexed -> 0-indexed

  # Step 2: Expand KV cache to beam_size
  kv_cache <- expand_kv_cache(kv_cache, beam_size)

  # Step 3: Expand encoder_output
  encoder_output_expanded <- encoder_output$`repeat`(c(beam_size, 1L, 1L))

  # Initialize beams: each has (tokens, cumulative_log_prob)
  beams <- lapply(seq_len(beam_size), function(b) {
    list(
      tokens = c(init_ids, top_token_ids[b]),
      cum_log_prob = top_log_probs[b]
    )
  })

  finished <- list()

  # Step 4: Beam search loop
  # Max steps = max_length - sample_begin - 1 (initial token already chosen)
  max_steps <- max_length - sample_begin - 1L
  torch::with_no_grad({
    for (step in seq_len(max_steps)) {
      if (length(beams) == 0) break
      if (length(finished) >= max_candidates) break
      # Stop if beams have reached max content length
      if (length(beams[[1]]$tokens) >= max_length) break

      n_active <- length(beams)

      # Build token tensor from last token of each beam (0-indexed IDs)
      last_tokens <- sapply(beams, function(b) b$tokens[length(b$tokens)])
      token_tensor <- torch::torch_tensor(
        matrix(last_tokens, ncol = 1L),
        dtype = torch::torch_long(), device = device)

      # If fewer active beams than beam_size, select relevant KV cache rows
      if (n_active < beam_size) {
        indices <- torch::torch_tensor(seq_len(n_active),
          dtype = torch::torch_long(), device = device)
        active_kv <- rearrange_kv_cache(kv_cache, indices, device)
        active_encoder <- encoder_output_expanded[1:n_active, , ]
        if (active_encoder$dim() == 2L) {
          active_encoder <- active_encoder$unsqueeze(1L)
        }
      } else {
        active_kv <- kv_cache
        active_encoder <- encoder_output_expanded
      }

      # Forward pass
      result <- model$decode(token_tensor, active_encoder,
        kv_cache = active_kv, need_weights = FALSE)
      new_kv_cache <- result$kv_cache
      logits <- result$logits

      # Get logits for last position
      next_logits <- logits[, logits$size(2), ]
      if (next_logits$dim() == 1L) {
        next_logits <- next_logits$unsqueeze(1L)
      }

      # Collect all candidates across beams
      all_candidates <- list()

      for (b in seq_len(n_active)) {
        beam_logits <- next_logits[b, ]$unsqueeze(1L)

        if (timestamps) {
          beam_logits <- apply_timestamp_rules(beam_logits,
            beams[[b]]$tokens, special, sample_begin)
        }

        beam_log_probs <- torch::nnf_log_softmax(beam_logits, dim = -1L)

        # Get top (beam_size + 1) candidates per beam
        n_cand <- min(beam_size + 1L, beam_log_probs$size(2))
        top <- torch::torch_topk(beam_log_probs$squeeze(1L), n_cand)
        cand_log_probs <- as.numeric(as.array(top[[1]]$cpu()))
        cand_token_ids <- as.integer(as.array(top[[2]]$cpu())) - 1L

        for (c_idx in seq_len(n_cand)) {
          all_candidates <- c(all_candidates, list(list(
            beam_idx = b,
            token_id = cand_token_ids[c_idx],
            cum_log_prob = beams[[b]]$cum_log_prob + cand_log_probs[c_idx],
            tokens = c(beams[[b]]$tokens, cand_token_ids[c_idx])
          )))
        }
      }

      # Sort by cumulative log prob (descending)
      scores <- sapply(all_candidates, function(c) c$cum_log_prob)
      order_idx <- order(scores, decreasing = TRUE)

      # Select top beam_size non-finished + move finished
      new_beams <- list()
      beam_source <- integer(0)

      for (idx in order_idx) {
        cand <- all_candidates[[idx]]

        if (cand$token_id == special$eot) {
          # Finished hypothesis (exclude EOT from tokens, keep its logprob in score)
          fin_tokens <- beams[[cand$beam_idx]]$tokens
          n_content <- length(fin_tokens) - sample_begin
          finished <- c(finished, list(list(
            tokens = fin_tokens,
            cum_log_prob = cand$cum_log_prob,
            n_tokens = n_content
          )))
          if (length(finished) >= max_candidates) break
        } else {
          if (length(new_beams) < beam_size) {
            new_beams <- c(new_beams, list(list(
              tokens = cand$tokens,
              cum_log_prob = cand$cum_log_prob
            )))
            beam_source <- c(beam_source, cand$beam_idx)
          }
        }

        if (length(new_beams) >= beam_size &&
            length(finished) >= max_candidates) break
      }

      if (length(new_beams) == 0) break

      # Rearrange KV cache for new beam ordering
      # Pad beam_source to beam_size if needed
      if (length(beam_source) < beam_size) {
        beam_source <- c(beam_source,
          rep(beam_source[1], beam_size - length(beam_source)))
      }
      source_indices <- torch::torch_tensor(beam_source,
        dtype = torch::torch_long(), device = device)
      kv_cache <- rearrange_kv_cache(new_kv_cache, source_indices, device)

      # Also re-expand encoder output if needed
      if (length(new_beams) <= beam_size) {
        encoder_output_expanded <- encoder_output$`repeat`(c(beam_size, 1L, 1L))
      }

      beams <- new_beams
    }
  })

  # If no finished hypotheses, use best active beam
  if (length(finished) == 0 && length(beams) > 0) {
    best <- beams[[1]]
    finished <- list(list(
      tokens = best$tokens,
      cum_log_prob = best$cum_log_prob,
      n_tokens = length(best$tokens) - sample_begin
    ))
  }

  # Score finished sequences with length penalty
  best_score <- -Inf
  best_hyp <- NULL
  for (hyp in finished) {
    # OpenAI Whisper length penalty: (5 + length) / 6) ^ alpha
    score <- hyp$cum_log_prob / ((5 + hyp$n_tokens) / 6) ^ length_penalty
    if (score > best_score) {
      best_score <- score
      best_hyp <- hyp
    }
  }

  # Word timestamps: re-run forced decode to collect cross-attention weights
  cross_attn_weights <- NULL
  if (word_timestamps) {
    cross_attn_weights <- forced_decode(model, encoder_output,
      best_hyp$tokens, device)
  }

  list(
    tokens = best_hyp$tokens,
    cross_attn_weights = cross_attn_weights,
    sum_logprob = best_hyp$cum_log_prob,
    n_tokens = best_hyp$n_tokens
  )
}

#' Decode with Temperature Fallback
#'
#' Try decoding at progressively higher temperatures until quality
#' thresholds are met. At temperature 0, uses beam search (or greedy
#' if beam_size=1). At temperature > 0, uses sampling with best-of.
#'
#' @param model WhisperModel
#' @param encoder_output Encoder hidden states
#' @param initial_tokens Initial token tensor
#' @param tokenizer Tokenizer
#' @param temperatures Numeric vector of temperatures to try
#' @param beam_size Number of beams for temp=0
#' @param best_of Number of samples for temp>0
#' @param max_length Maximum output length
#' @param timestamps Whether to allow timestamp tokens
#' @param word_timestamps Whether to collect cross-attention weights
#' @param compression_ratio_threshold Max compression ratio
#' @param logprob_threshold Min average log probability
#' @param length_penalty Length penalty for beam search
#' @param patience Patience factor for beam search
#' @param device Device
#' @return List with tokens, cross_attn_weights, sum_logprob, n_tokens
decode_with_fallback <- function(
  model,
  encoder_output,
  initial_tokens,
  tokenizer,
  temperatures = c(0, 0.2, 0.4, 0.6, 0.8, 1.0),
  beam_size = 5L,
  best_of = 5L,
  max_length = 448L,
  timestamps = FALSE,
  word_timestamps = FALSE,
  compression_ratio_threshold = 2.4,
  logprob_threshold = -1.0,
  length_penalty = 1.0,
  patience = Inf,
  device
) {
  for (temp in temperatures) {
    if (temp == 0) {
      if (beam_size > 1L) {
        decode_result <- beam_search_decode(model, encoder_output,
          initial_tokens, tokenizer,
          beam_size = beam_size, max_length = max_length,
          timestamps = timestamps, word_timestamps = word_timestamps,
          length_penalty = length_penalty, patience = patience,
          device = device)
      } else {
        decode_result <- greedy_decode(model, encoder_output,
          initial_tokens, tokenizer,
          max_length = max_length,
          timestamps = timestamps, word_timestamps = word_timestamps,
          device = device)
      }
    } else {
      # Sample best_of times, keep best by average log prob
      best_result <- NULL
      best_avg <- -Inf

      for (s in seq_len(best_of)) {
        candidate <- sample_decode(model, encoder_output,
          initial_tokens, tokenizer,
          temperature = temp, max_length = max_length,
          timestamps = timestamps, word_timestamps = word_timestamps,
          device = device)

        avg_lp <- if (candidate$n_tokens > 0) {
          candidate$sum_logprob / candidate$n_tokens
        } else {
          -Inf
        }

        if (avg_lp > best_avg) {
          best_avg <- avg_lp
          best_result <- candidate
        }
      }

      decode_result <- best_result
    }

    # Check quality thresholds
    sample_begin <- length(as.integer(as.array(initial_tokens$cpu())))
    content_tokens <- decode_result$tokens[seq_len(length(decode_result$tokens)) > sample_begin]
    text <- tokenizer$decode(content_tokens)

    # Compression ratio check (skip for very short text)
    if (nchar(text) > 0) {
      cr <- compression_ratio(text)
      if (cr > compression_ratio_threshold) next
    }

    # Average log prob check
    if (decode_result$n_tokens > 0) {
      avg_logprob <- decode_result$sum_logprob / decode_result$n_tokens
      if (avg_logprob < logprob_threshold) next
    }

    # Quality OK
    return(decode_result)
  }

  # All temperatures failed, return last result
  decode_result
}

