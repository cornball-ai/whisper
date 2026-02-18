# Package-level cache for expensive lookups (special tokens, added_tokens.json)
.whisper_cache <- new.env(parent = emptyenv())

#' Whisper Model Configurations
#'
#' Get configuration for a Whisper model variant.
#'
#' @param model Character. Model name: "tiny", "base", "small", "medium", "large-v3"
#' @return List with model configuration parameters
#' @export
#' @examples
#' # Get tiny model configuration
#' cfg <- whisper_config("tiny")
#' cfg$n_mels
#' cfg$n_audio_layer
#'
#' # Compare model sizes
#' whisper_config("tiny")$n_text_layer
#' whisper_config("large-v3")$n_text_layer
whisper_config <- function(model = "tiny") {
  configs <- list(
    tiny = list(
      n_mels = 80L,
      n_audio_ctx = 1500L,
      n_audio_state = 384L,
      n_audio_head = 6L,
      n_audio_layer = 4L,
      n_vocab = 51865L,
      n_text_ctx = 448L,
      n_text_state = 384L,
      n_text_head = 6L,
      n_text_layer = 4L,
      hf_repo = "openai/whisper-tiny",
      # (layer, head) pairs for cross-attention alignment (0-indexed)
      alignment_heads = matrix(c(
        1, 0, 2, 0, 2, 5, 3, 0, 3, 1, 3, 2, 3, 3, 3, 4
      ), ncol = 2, byrow = TRUE)
    ),
    base = list(
      n_mels = 80L,
      n_audio_ctx = 1500L,
      n_audio_state = 512L,
      n_audio_head = 8L,
      n_audio_layer = 6L,
      n_vocab = 51865L,
      n_text_ctx = 448L,
      n_text_state = 512L,
      n_text_head = 8L,
      n_text_layer = 6L,
      hf_repo = "openai/whisper-base",
      alignment_heads = matrix(c(
        3, 1, 4, 2, 4, 3, 4, 7, 5, 1, 5, 2, 5, 4, 5, 6
      ), ncol = 2, byrow = TRUE)
    ),
    small = list(
      n_mels = 80L,
      n_audio_ctx = 1500L,
      n_audio_state = 768L,
      n_audio_head = 12L,
      n_audio_layer = 12L,
      n_vocab = 51865L,
      n_text_ctx = 448L,
      n_text_state = 768L,
      n_text_head = 12L,
      n_text_layer = 12L,
      hf_repo = "openai/whisper-small",
      alignment_heads = matrix(c(
        6, 6, 7, 0, 7, 3, 7, 8, 8, 2, 8, 5, 8, 7, 9, 0, 9, 4, 9, 8,
        9, 10, 10, 0, 10, 1, 10, 2, 10, 3, 10, 6, 10, 11, 11, 2, 11, 4
      ), ncol = 2, byrow = TRUE)
    ),
    medium = list(
      n_mels = 80L,
      n_audio_ctx = 1500L,
      n_audio_state = 1024L,
      n_audio_head = 16L,
      n_audio_layer = 24L,
      n_vocab = 51865L,
      n_text_ctx = 448L,
      n_text_state = 1024L,
      n_text_head = 16L,
      n_text_layer = 24L,
      hf_repo = "openai/whisper-medium",
      alignment_heads = matrix(c(
        13, 15, 15, 4, 15, 15, 16, 1, 20, 0, 23, 4
      ), ncol = 2, byrow = TRUE)
    ),
    `large-v3` = list(
      n_mels = 128L,
      n_audio_ctx = 1500L,
      n_audio_state = 1280L,
      n_audio_head = 20L,
      n_audio_layer = 32L,
      n_vocab = 51866L,
      n_text_ctx = 448L,
      n_text_state = 1280L,
      n_text_head = 20L,
      n_text_layer = 32L,
      hf_repo = "openai/whisper-large-v3",
      alignment_heads = matrix(c(
        9, 19, 11, 2, 11, 4, 11, 17, 22, 7, 22, 11, 22, 17, 23, 2, 23, 15
      ), ncol = 2, byrow = TRUE)
    )
  )

  if (!model %in% names(configs)) {
    stop("Unknown model: ", model, ". Choose from: ",
      paste(names(configs), collapse = ", "))
  }

  cfg <- configs[[model]]
  cfg$model_name <- model
  cfg
}

#' Special Token IDs
#'
#' Get special token IDs for a Whisper model.
#' Token IDs differ between model variants (e.g., large-v3 has extra language tokens).
#'
#' @param model Model name (default: "tiny")
#' @return Named list of special token IDs
whisper_special_tokens <- function(model = "tiny") {
  cache_key <- paste0(".special_tokens_", model)
  cached <- .whisper_cache[[cache_key]]
  if (!is.null(cached)) return(cached)

  # Load from model's added_tokens.json for accuracy
  cfg <- whisper_config(model)
  added_tokens <- load_added_tokens(cfg$hf_repo)

  # Helper to get token with fallback
  get_token <- function(
    name,
    default
  ) {
    if (!is.null(added_tokens) && name %in% names(added_tokens)) {
      as.integer(added_tokens[[name]])
    } else {
      default
    }
  }

  # Extract special tokens, using fallbacks for tokens not in added_tokens.json
  # (Some models have tokens in vocab.json instead of added_tokens.json)
  result <- list(
    sot = get_token("<|startoftranscript|>", 50258L),
    eot = get_token("<|endoftext|>", 50257L),
    translate = get_token("<|translate|>", 50358L),
    transcribe = get_token("<|transcribe|>", 50359L),
    no_speech = get_token("<|nospeech|>", 50362L),
    no_timestamps = get_token("<|notimestamps|>", 50363L),
    timestamp_begin = get_token("<|0.00|>", 50364L),
    lang_en = get_token("<|en|>", 50259L)
  )
  .whisper_cache[[cache_key]] <- result
  result
}

#' Load Added Tokens from HuggingFace
#'
#' @param repo HuggingFace repo ID
#' @return Named list of token -> ID mappings, or NULL if not found
load_added_tokens <- function(repo) {
  cache_key <- paste0(".added_tokens_", repo)
  cached <- .whisper_cache[[cache_key]]
  if (!is.null(cached)) return(cached)

  result <- tryCatch({
      path <- hfhub::hub_download(repo, "added_tokens.json")
      jsonlite::fromJSON(path)
    }, error = function(e) {
      NULL
    })
  if (!is.null(result)) .whisper_cache[[cache_key]] <- result
  result
}

#' Get Language Token ID
#'
#' @param lang Two-letter language code (e.g., "en", "es", "fr")
#' @param model Model name for correct token IDs
#' @return Token ID for the language
whisper_lang_token <- function(
  lang = "en",
  model = "tiny"
) {
  # Load from model's added_tokens.json for accuracy
  cfg <- whisper_config(model)
  added_tokens <- load_added_tokens(cfg$hf_repo)

  if (!is.null(added_tokens)) {
    # Look up language token directly
    token_name <- paste0("<|", lang, "|>")
    if (token_name %in% names(added_tokens)) {
      return(as.integer(added_tokens[[token_name]]))
    }
  }

  # Fallback to offset calculation (works for tiny/base/small/medium)
  langs <- c(
    en = 0L, zh = 1L, de = 2L, es = 3L, ru = 4L, ko = 5L, fr = 6L,
    ja = 7L, pt = 8L, tr = 9L, pl = 10L, ca = 11L, nl = 12L, ar = 13L,
    sv = 14L, it = 15L, id = 16L, hi = 17L, fi = 18L, vi = 19L,
    he = 20L, uk = 21L, el = 22L, ms = 23L, cs = 24L, ro = 25L,
    da = 26L, hu = 27L, ta = 28L, no = 29L, th = 30L, ur = 31L,
    hr = 32L, bg = 33L, lt = 34L, la = 35L, mi = 36L, ml = 37L,
    cy = 38L, sk = 39L, te = 40L, fa = 41L, lv = 42L, bn = 43L,
    sr = 44L, az = 45L, sl = 46L, kn = 47L, et = 48L, mk = 49L,
    br = 50L, eu = 51L, is = 52L, hy = 53L, ne = 54L, mn = 55L,
    bs = 56L, kk = 57L, sq = 58L, sw = 59L, gl = 60L, mr = 61L,
    pa = 62L, si = 63L, km = 64L, sn = 65L, yo = 66L, so = 67L,
    af = 68L, oc = 69L, ka = 70L, be = 71L, tg = 72L, sd = 73L,
    gu = 74L, am = 75L, yi = 76L, lo = 77L, uz = 78L, fo = 79L,
    ht = 80L, ps = 81L, tk = 82L, nn = 83L, mt = 84L, sa = 85L,
    lb = 86L, my = 87L, bo = 88L, tl = 89L, mg = 90L, as = 91L,
    tt = 92L, haw = 93L, ln = 94L, ha = 95L, ba = 96L, jw = 97L,
    su = 98L
  )

  if (!lang %in% names(langs)) {
    stop("Unknown language: ", lang)
  }

  50259L + langs[[lang]]
}

