#' Whisper Transcription
#'
#' Main transcription API for Whisper.

#' Transcribe Audio
#'
#' Transcribe speech from an audio file using Whisper.
#'
#' @param file Path to audio file (WAV, MP3, etc.)
#' @param model Model name: "tiny", "base", "small", "medium", "large-v3"
#' @param language Language code (e.g., "en", "es"). NULL for auto-detection.
#' @param task "transcribe" or "translate" (translate to English)
#' @param device Device: "auto", "cpu", "cuda"
#' @param dtype Data type: "auto", "float16", "float32"
#' @param verbose Print progress messages
#' @return List with text, language, and metadata
#' @export
#' @examples
#' \donttest{
#' # Transcribe included sample (JFK "ask not" speech)
#' if (model_exists("tiny")) {
#'   audio_file <- system.file("audio", "jfk.mp3", package = "whisper")
#'   result <- transcribe(audio_file, model = "tiny")
#'   result$text
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
  device = "auto",
  dtype = "auto",
  verbose = TRUE
) {
  if (!file.exists(file)) {
    stop("Audio file not found: ", file)
  }

  device <- parse_device(device)
  dtype <- parse_dtype(dtype, device)

  # Load model (prompt to download if needed)
  if (verbose) message("Loading model: ", model)
  whisper <- load_whisper_model(model, device = device, dtype = dtype,
    download = TRUE, verbose = verbose)

  # Load tokenizer
  tokenizer <- whisper_tokenizer(model)

  # Get model config
  config <- whisper_config(model)

  # Check audio duration
  duration <- audio_duration(file)
  if (verbose) message("Audio duration: ", round(duration, 1), "s")

  # Process audio
  if (duration <= WHISPER_CHUNK_LENGTH) {
    # Single chunk processing
    result <- transcribe_chunk(file, whisper, tokenizer, config,
      language = language, task = task,
      device = device, dtype = dtype, verbose = verbose)
  } else {
    # Long audio - process in chunks
    result <- transcribe_long(file, whisper, tokenizer, config,
      language = language, task = task,
      device = device, dtype = dtype, verbose = verbose)
  }

  # Add metadata
  result$model <- model
  result$backend <- "whisper"
  result$duration <- duration

  result
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
  device,
  dtype,
  verbose = TRUE
) {
  # Convert audio to mel spectrogram
  if (verbose) message("Processing audio...")
  mel <- audio_to_mel(file, n_mels = config$n_mels, device = device, dtype = dtype)

  # Get initial decoder tokens (use model name for correct special token IDs)
  initial_tokens <- get_initial_tokens(language, task, model = config$model_name)
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
  generated <- greedy_decode(model, encoder_output, tokens, tokenizer,
    max_length = config$n_text_ctx,
    device = device)

  # Decode tokens to text
  text <- tokenizer$decode(generated)

  # Clean up text
  text <- clean_text(text)

  # Build result
  list(
    text = text,
    language = language
  )
}

#' Greedy Decoding
#'
#' @param model WhisperModel
#' @param encoder_output Encoder hidden states
#' @param initial_tokens Initial token tensor
#' @param tokenizer Tokenizer
#' @param max_length Maximum output length
#' @param device Device
#' @return Integer vector of generated tokens
greedy_decode <- function(
  model,
  encoder_output,
  initial_tokens,
  tokenizer,
  max_length = 448L,
  device
) {
  # Use model-specific special tokens
  special <- whisper_special_tokens(tokenizer$model)
  generated <- as.integer(as.array(initial_tokens$cpu()))

  kv_cache <- NULL
  tokens <- initial_tokens

  torch::with_no_grad({
      for (i in seq_len(max_length)) {
        # Stop if we've reached max context length
        if (length(generated) >= max_length) break

        # Get next token logits
        result <- model$decode(tokens, encoder_output, kv_cache = kv_cache)
        logits <- result$logits
        kv_cache <- result$kv_cache

        # Get last position logits (R uses 1-based indexing, not negative indexing like Python)
        seq_len <- logits$size(2)
        next_logits <- logits[, seq_len,]# (batch, vocab)

        # Greedy: take argmax (subtract 1 because R torch argmax returns 1-indexed)
        next_token <- next_logits$argmax(dim = - 1L)
        next_token_id <- as.integer(next_token$item()) - 1L

        # Check for end of text
        if (next_token_id == special$eot) {
          break
        }

        # Append token
        generated <- c(generated, next_token_id)

        # Prepare next input (decoder expects 0-indexed token IDs, adds 1 internally)
        tokens <- torch::torch_tensor(matrix(next_token_id, nrow = 1L),
          dtype = torch::torch_long(),
          device = device)
      }
    })

  generated
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
  device,
  dtype,
  verbose
) {
  # Split into chunks
  chunks <- split_audio(file, chunk_length = 30, overlap = 1)

  if (verbose) message("Processing ", length(chunks), " chunks...")

  all_text <- character(length(chunks))

  for (i in seq_along(chunks)) {
    if (verbose) message("  Chunk ", i, "/", length(chunks))

    # Convert chunk to mel
    mel <- audio_to_mel(chunks[[i]], n_mels = config$n_mels,
      device = device, dtype = dtype)

    # Get initial tokens (use model name for correct special token IDs)
    initial_tokens <- get_initial_tokens(language, task, model = config$model_name)
    tokens <- torch::torch_tensor(matrix(initial_tokens, nrow = 1),
      dtype = torch::torch_long(),
      device = device)

    # Encode
    torch::with_no_grad({
        encoder_output <- model$encode(mel)
      })

    # Decode
    generated <- greedy_decode(model, encoder_output, tokens, tokenizer,
      max_length = config$n_text_ctx,
      device = device)

    # Decode to text
    text <- tokenizer$decode(generated)
    text <- clean_text(text)
    all_text[i] <- text
  }

  # Combine results
  list(
    text = paste(all_text, collapse = " "),
    language = language
  )
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

