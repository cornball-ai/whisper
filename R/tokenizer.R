#' Whisper BPE Tokenizer
#'
#' Byte-pair encoding tokenizer for Whisper models.

#' Create Whisper Tokenizer
#'
#' Load or create a Whisper tokenizer from HuggingFace vocab files.
#'
#' @param model Model name for vocab lookup
#' @param cache_dir Directory for cached vocab files
#' @return Tokenizer object (list with encode/decode functions)
#' @export
whisper_tokenizer <- function(
  model = "tiny",
  cache_dir = getOption("whisper.cache_dir")
) {
  # Ensure vocab files are downloaded
  vocab_dir <- ensure_tokenizer_files(model, cache_dir)

  # Load vocab and merges
  vocab_file <- file.path(vocab_dir, "vocab.json")
  merges_file <- file.path(vocab_dir, "merges.txt")

  vocab <- jsonlite::fromJSON(vocab_file)
  merges_text <- readLines(merges_file, warn = FALSE)

  # Skip header line if present
  if (length(merges_text) > 0 && grepl("^#", merges_text[1])) {
    merges_text <- merges_text[- 1]
  }

  # Parse merges into list of pairs
  merges <- lapply(merges_text, function(line) {
      parts <- strsplit(line, " ", fixed = TRUE) [[1]]
      if (length(parts) == 2) parts else NULL
    })
  merges <- Filter(Negate(is.null), merges)

  # Create merge ranking (lower = higher priority)
  merge_ranks <- setNames(seq_along(merges), sapply(merges, paste, collapse = " "))

  # Create reverse vocab for decoding
  id_to_token <- setNames(names(vocab), as.character(unlist(vocab)))

  # Get special tokens (using model-specific IDs)
  special <- whisper_special_tokens(model)

  structure(
    list(
      vocab = vocab,
      id_to_token = id_to_token,
      merges = merges,
      merge_ranks = merge_ranks,
      special_tokens = special,
      model = model,
      encode = function(text) tokenizer_encode(text, vocab, merge_ranks),
      decode = function(ids) tokenizer_decode(ids, id_to_token, special),
      encode_special = function(token) vocab[[token]],
      n_vocab = length(vocab)
    ),
    class = "whisper_tokenizer"
  )
}

#' Encode Text to Token IDs
#'
#' @param text Character string to encode
#' @param vocab Vocabulary mapping (token -> id)
#' @param merge_ranks Merge ranking for BPE
#' @return Integer vector of token IDs
tokenizer_encode <- function(
  text,
  vocab,
  merge_ranks
) {
  if (is.null(text) || text == "") {
    return(integer(0))
  }

  # Convert text to bytes (UTF-8)
  bytes <- charToRaw(text)

  # Convert bytes to initial tokens (byte-level BPE)
  # Whisper uses GPT-2 byte encoding
  tokens <- sapply(bytes, function(b) {
      byte_token <- byte_to_token(as.integer(b))
      byte_token
    }, USE.NAMES = FALSE)

  # Apply BPE merges iteratively
  tokens <- apply_bpe(tokens, merge_ranks)

  # Convert tokens to IDs
  ids <- sapply(tokens, function(t) {
      if (t %in% names(vocab)) {
        vocab[[t]]
      } else {
        # Unknown token - try to find closest match or use special
        vocab[["<|endoftext|>"]]# fallback
      }
    }, USE.NAMES = FALSE)

  as.integer(ids)
}

#' Convert Byte to BPE Token
#'
#' GPT-2/Whisper uses a specific byte-to-unicode mapping.
#'
#' @param byte Integer byte value (0-255)
#' @return Character token
byte_to_token <- function(byte) {
  # GPT-2 byte encoder mapping
  # Printable ASCII (33-126) + some others map to themselves
  # Others map to 256+ unicode codepoints

  if (byte >= 33 && byte <= 126) {
    # Printable ASCII (except space)
    intToUtf8(byte)
  } else if (byte == 32) {
    # Space maps to special char
    "\u0120"# Ġ
  } else if (byte >= 161 && byte <= 172) {
    intToUtf8(byte)
  } else if (byte >= 174 && byte <= 255) {
    intToUtf8(byte)
  } else {
    # Map unprintable bytes to 256+ unicode range
    intToUtf8(256 + byte)
  }
}

#' Apply BPE Merges
#'
#' @param tokens Character vector of tokens
#' @param merge_ranks Named vector of merge rankings
#' @return Character vector after BPE merges
apply_bpe <- function(
  tokens,
  merge_ranks
) {
  if (length(tokens) <= 1) {
    return(tokens)
  }

  while (TRUE) {
    # Find best merge (lowest rank)
    best_merge <- NULL
    best_rank <- Inf
    best_idx <- NULL

    for (i in seq_len(length(tokens) - 1)) {
      pair <- paste(tokens[i], tokens[i + 1])
      if (pair %in% names(merge_ranks)) {
        rank <- merge_ranks[[pair]]
        if (rank < best_rank) {
          best_rank <- rank
          best_merge <- pair
          best_idx <- i
        }
      }
    }

    if (is.null(best_merge)) {
      break
    }

    # Apply merge
    merged_token <- paste0(tokens[best_idx], tokens[best_idx + 1])
    tokens <- c(
      if (best_idx > 1) tokens[1:(best_idx - 1)] else character(0),
      merged_token,
      if (best_idx + 2 <= length(tokens)) tokens[(best_idx + 2) :length(tokens)] else character(0)
    )
  }

  tokens
}

#' Decode Token IDs to Text
#'
#' @param ids Integer vector of token IDs
#' @param id_to_token Mapping from ID to token
#' @param special_tokens Special token info
#' @return Character string
tokenizer_decode <- function(
  ids,
  id_to_token,
  special_tokens
) {
  # Filter out special tokens (optionally)
  special_ids <- unlist(special_tokens)

  tokens <- sapply(ids, function(id) {
      id_str <- as.character(id)
      if (id_str %in% names(id_to_token)) {
        id_to_token[[id_str]]
      } else {
        ""
      }
    }, USE.NAMES = FALSE)

  # Join tokens
  text <- paste(tokens, collapse = "")

  # Decode byte-level BPE back to text
  text <- decode_bpe_bytes(text)

  text
}

#' Decode BPE Bytes Back to Text
#'
#' @param text Text with BPE byte tokens
#' @return Decoded text
decode_bpe_bytes <- function(text) {
  # Replace special space token
  text <- gsub("\u0120", " ", text, fixed = TRUE)

  # Handle other byte-level encodings
  # This is a simplified version - full implementation would

  # reverse the byte_to_token mapping

  text
}

#' Ensure Tokenizer Files are Downloaded
#'
#' @param model Model name
#' @param cache_dir Cache directory (unused, kept for API compatibility)
#' @return Path to vocab directory (directory containing vocab.json)
ensure_tokenizer_files <- function(
  model,
  cache_dir
) {
  cfg <- whisper_config(model)
  repo <- cfg$hf_repo

  # Download if not present, get path from hfhub
  vocab_file <- hfhub::hub_download(repo, "vocab.json")
  hfhub::hub_download(repo, "merges.txt")

  # Return directory containing vocab files
  dirname(vocab_file)
}

#' Download Tokenizer Files from HuggingFace
#'
#' @param model Model name
#' @param vocab_dir Directory to save files (unused, kept for API compatibility)
download_tokenizer_files <- function(
  model,
  vocab_dir
) {
  cfg <- whisper_config(model)
  repo <- cfg$hf_repo

  message("Downloading tokenizer files for ", model, " via hfhub...")

  hfhub::hub_download(repo, "vocab.json")
  hfhub::hub_download(repo, "merges.txt")

  message("Tokenizer files downloaded")
}

#' Get Initial Decoder Tokens
#'
#' Build the initial token sequence for decoder input.
#'
#' @param language Two-letter language code or NULL for auto
#' @param task "transcribe" or "translate"
#' @param timestamps Whether to include timestamps
#' @param model Model name for correct special token IDs
#' @return Integer vector of initial token IDs
get_initial_tokens <- function(
  language = "en",
  task = "transcribe",
  timestamps = TRUE,
  model = "tiny"
) {
  special <- whisper_special_tokens(model)

  tokens <- c(special$sot)

  # Add language token if specified
  if (!is.null(language)) {
    tokens <- c(tokens, whisper_lang_token(language, model))
  }

  # Add task token
  if (task == "transcribe") {
    tokens <- c(tokens, special$transcribe)
  } else if (task == "translate") {
    tokens <- c(tokens, special$translate)
  }

  # Add timestamp token
  if (!timestamps) {
    tokens <- c(tokens, special$no_timestamps)
  }

  as.integer(tokens)
}

#' Check if Token is Timestamp
#'
#' @param token_id Token ID
#' @param model Model name for correct token IDs
#' @return TRUE if timestamp token
is_timestamp_token <- function(
  token_id,
  model = "tiny"
) {
  special <- whisper_special_tokens(model)
  token_id >= special$timestamp_begin
}

#' Decode Timestamp Token
#'
#' @param token_id Token ID
#' @param model Model name for correct token IDs
#' @return Time in seconds
decode_timestamp <- function(
  token_id,
  model = "tiny"
) {
  special <- whisper_special_tokens(model)
  if (token_id < special$timestamp_begin) {
    return(NA_real_)
  }
  # Each timestamp token represents 0.02 seconds
  (token_id - special$timestamp_begin) * 0.02
}

