#' Whisper BPE Tokenizer
#'
#' Byte-pair encoding tokenizer for Whisper models.

#' Create Whisper Tokenizer
#'
#' Load or create a Whisper tokenizer from HuggingFace vocab files.
#'
#' @param model Model name for vocab lookup
#' @return Tokenizer object (list with encode/decode functions)
#' @export
#' @examples
#' \donttest{
#' # Load tokenizer (requires prior model download)
#' if (model_exists("tiny")) {
#'   tok <- whisper_tokenizer("tiny")
#'   tok$encode("Hello world")
#'   tok$decode(c(50258, 50259, 50359, 50363))
#' }
#' }
whisper_tokenizer <- function(model = "tiny") {
  # Ensure vocab files are downloaded
  vocab_dir <- ensure_tokenizer_files(model)

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

#' Build Reverse Byte Decoder
#'
#' Inverts the GPT-2 byte-to-unicode mapping used by byte_to_token().
#' Cached after first call.
#'
#' @return Named character vector mapping unicode codepoint (as string) to
#'   raw byte value
build_byte_decoder <- function() {
  if (!is.null(.tokenizer_cache$byte_decoder)) {
    return(.tokenizer_cache$byte_decoder)
  }
  decoder <- integer(256)
  names(decoder) <- character(256)
  for (b in 0:255) {
    cp <- utf8ToInt(byte_to_token(b))
    names(decoder)[b + 1L] <- as.character(cp)
    decoder[b + 1L] <- b
  }
  .tokenizer_cache$byte_decoder <- decoder
  decoder
}

# Module-level cache for byte decoder
.tokenizer_cache <- new.env(parent = emptyenv())

#' Decode BPE Bytes Back to Text
#'
#' Reverses the GPT-2 byte-level encoding, converting unicode tokens
#' back to raw UTF-8 bytes.
#'
#' @param text Text with BPE byte tokens
#' @return Decoded UTF-8 text
decode_bpe_bytes <- function(text) {
  if (nchar(text) == 0) return(text)

  decoder <- build_byte_decoder()
  codepoints <- utf8ToInt(text)
  bytes <- raw(length(codepoints))

  for (i in seq_along(codepoints)) {
    cp_str <- as.character(codepoints[i])
    idx <- match(cp_str, names(decoder))
    if (!is.na(idx)) {
      bytes[i] <- as.raw(decoder[idx])
    } else {
      bytes[i] <- charToRaw("?")
    }
  }

  # Write raw bytes to a connection and read back as UTF-8,

  # replacing any invalid multibyte sequences
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)
  writeBin(bytes, tmp)
  out <- readLines(tmp, warn = FALSE, encoding = "UTF-8")
  out <- paste(out, collapse = "\n")
  # Strip any remaining invalid bytes
  iconv(out, from = "UTF-8", to = "UTF-8", sub = "")
}

#' Ensure Tokenizer Files are Downloaded
#'
#' @param model Model name
#' @return Path to vocab directory (directory containing vocab.json)
ensure_tokenizer_files <- function(model) {
  cfg <- whisper_config(model)
  repo <- cfg$hf_repo

  # Check if files exist locally (do NOT download without consent)
  vocab_file <- tryCatch(
    hfhub::hub_download(repo, "vocab.json", local_files_only = TRUE),
    error = function(e) NULL
  )
  merges_file <- tryCatch(
    hfhub::hub_download(repo, "merges.txt", local_files_only = TRUE),
    error = function(e) NULL
  )

  if (is.null(vocab_file) || is.null(merges_file)) {
    stop(
      "Tokenizer files not found for model '", model, "'. ",
      "Run download_whisper_model('", model, "') first.",
      call. = FALSE
    )
  }

  # Return directory containing vocab files
  dirname(vocab_file)
}

#' Download Tokenizer Files from HuggingFace
#'
#' @param model Model name
download_tokenizer_files <- function(model) {
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
#' @param model Model name for correct special token IDs
#' @param timestamps Whether to include timestamps (internal use)
#' @return Integer vector of initial token IDs
get_initial_tokens <- function(
  language = "en",
  task = "transcribe",
  model = "tiny",
  timestamps = FALSE
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

