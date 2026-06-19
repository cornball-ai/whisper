#' Logit Suppression Filters
#'
#' Faithful port of the suppression logit filters from openai-whisper
#' (\code{whisper/decoding.py} and \code{whisper/tokenizer.py}). Two filters
#' run during decoding:
#'
#' \itemize{
#'   \item \code{SuppressTokens}: at every step, drive a fixed set of token
#'     logits to \code{-Inf}. The set is the tokenizer's \emph{non-speech}
#'     tokens (brackets, symbols, music notes) plus the control tokens, which
#'     stops the model emitting speaker tags / non-speech annotations such as
#'     \code{[BLANK_AUDIO]}, \code{(MUSIC PLAYING)}, or \code{[DAVID]} while
#'     keeping ordinary punctuation.
#'   \item \code{SuppressBlank}: at the first generated step only, suppress a
#'     leading space and end-of-text so decoding does not start with blank.
#' }
#'
#' These are independent of the timestamp rules and the \code{no_speech_prob}
#' silent-window skip; they apply regardless of whether timestamps are on.

#' Non-speech token IDs
#'
#' Mirrors \code{whisper/tokenizer.py::non_speech_tokens}. Encodes a fixed set
#' of symbols (and their space-prefixed forms); a symbol contributes its first
#' token when it encodes to a single token, or always for the musical symbols
#' (whose 3-byte UTF-8 forms share a leading token).
#'
#' @param encode The tokenizer's \code{encode} function (text -> 0-indexed IDs).
#' @return Sorted integer vector of 0-indexed token IDs.
#' @keywords internal
.non_speech_token_ids <- function(encode) {
  # list('"#()*+/:;<=>@[\]^_`{|}~') plus the CJK corner brackets
  indiv <- strsplit('"#()*+/:;<=>@[\\]^_`{|}~', "")[[1]]
  corner <- intToUtf8(c(0x300C, 0x300D, 0x300E, 0x300F), multiple = TRUE)
  # the .split() list, including (' (" and the music-note runs. The note runs
  # are built with intToUtf8 so this source file stays ASCII (CRAN portability).
  multi <- c(
    strsplit("<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }}",
      " ")[[1]],
    intToUtf8(rep(0x266A, 2L)), intToUtf8(rep(0x266A, 3L)))
  musical <- intToUtf8(c(0x2669, 0x266A, 0x266B, 0x266C, 0x266D, 0x266E, 0x266F),
    multiple = TRUE)
  symbols <- c(indiv, corner, multi)

  # allow "-" and "'" between words: suppress only their word-initial token
  result <- c(encode(" -")[1], encode(" '")[1])
  for (sym in c(symbols, musical)) {
    for (toks in list(encode(sym), encode(paste0(" ", sym)))) {
      if (length(toks) == 1L || sym %in% musical) {
        result <- c(result, toks[1])
      }
    }
  }
  sort(unique(as.integer(result)))
}

#' Token IDs suppressed at every decode step
#'
#' \code{non_speech_tokens} plus the control tokens the reference also adds in
#' \code{decoding.py::_get_suppress_tokens}: translate, transcribe, sot_lm,
#' sot_prev, sot, and no_speech.
#'
#' @param encode The tokenizer's \code{encode} function.
#' @param special Named list of special token IDs.
#' @return Sorted integer vector of 0-indexed token IDs.
#' @keywords internal
.decode_suppress_ids <- function(encode, special) {
  ctrl <- c(special$translate, special$transcribe,
    special$sot_lm, special$sot_prev,
    special$sot, special$no_speech)
  sort(unique(as.integer(c(.non_speech_token_ids(encode), ctrl))))
}

#' Token IDs suppressed at the first decode step (SuppressBlank)
#'
#' @param encode The tokenizer's \code{encode} function.
#' @param special Named list of special token IDs.
#' @return Sorted integer vector of 0-indexed token IDs.
#' @keywords internal
.blank_token_ids <- function(encode, special) {
  sort(unique(as.integer(c(encode(" "), special$eot))))
}

#' No-speech probability from the prompt prefill
#'
#' Mirrors \code{decoding.py}: the probability mass on the \code{<|nospeech|>}
#' token in the softmax taken at the SOT position of the prefill logits (the
#' distribution that chooses the language slot). High values mean the window is
#' likely silence.
#'
#' @param logits Prefill logits, shape \code{(1, n_prompt, vocab)}.
#' @param generated Integer vector of the prompt tokens (to locate SOT).
#' @param special Named list of special token IDs.
#' @return Scalar no-speech probability (numeric).
#' @keywords internal
.no_speech_prob <- function(logits, generated, special) {
  sot_pos <- match(special$sot, generated)
  if (is.na(sot_pos)) {
    sot_pos <- 1L
  }
  probs <- torch::nnf_softmax(logits[1, sot_pos, ], dim = -1L)
  as.numeric(probs[special$no_speech + 1L]$item())
}

#' Build an additive logit mask (-Inf at suppressed positions, 0 elsewhere)
#'
#' Added to a \code{(1, n_vocab)} logit row before argmax. Using an additive
#' mask avoids tensor advanced-index assignment and works on any device/dtype.
#'
#' @param ids0 0-indexed token IDs to suppress.
#' @param n_vocab Vocabulary size (logit width).
#' @param device torch device.
#' @param dtype torch dtype.
#' @return A \code{(1, n_vocab)} tensor.
#' @keywords internal
.suppress_mask <- function(ids0, n_vocab, device, dtype) {
  v <- numeric(n_vocab)
  ids1 <- ids0[ids0 >= 0L & ids0 < n_vocab] + 1L
  v[ids1] <- -Inf
  torch::torch_tensor(v, device = device, dtype = dtype)$view(c(1L, n_vocab))
}
