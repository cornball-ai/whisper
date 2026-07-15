# anvl/yunque end-to-end transcription. Greedy decode without a KV cache:
# recompute the decoder over the growing prompt each step (O(S^2); correct
# and simple -- the KV cache is the optimization). Suppress masks and the EOT
# stop match the torch greedy path.

yq_greedy_decode <- function(xa, dec_w, tok, eot, prompt, max_length = 224L) {
  cross_kv <- .yq_cross_kv(xa, dec_w)
  suppress <- tok$suppress_tokens
  blank <- tok$blank_tokens
  # prefill the prompt (builds the initial self-KV cache)
  res <- yq_decoder_incremental(matrix(as.integer(prompt), 1L), cross_kv,
    NULL, dec_w, 0L)
  self_kv <- res$self_kv
  nl <- as.array(res$logits)[1L, length(prompt), ]
  nl[suppress + 1L] <- -Inf
  nl[blank + 1L] <- -Inf # SuppressBlank on the first generated step
  nxt <- which.max(nl) - 1L
  generated <- integer(0)
  offset <- length(prompt)
  for (step in seq_len(max_length)) {
    if (nxt == eot) {
      break
    }
    generated <- c(generated, nxt)
    res <- yq_decoder_incremental(matrix(nxt, 1L), cross_kv, self_kv, dec_w,
      offset)
    self_kv <- res$self_kv
    offset <- offset + 1L
    nl <- as.array(res$logits)[1L, 1L, ]
    nl[suppress + 1L] <- -Inf
    nxt <- which.max(nl) - 1L
  }
  generated
}

#' End-to-end transcription (anvl)
#'
#' Torch-free Whisper transcription on the anvl/XLA stack: mel frontend
#' (\code{\link{yq_audio_to_mel}}) -> encoder (\code{\link{yq_encoder}}) ->
#' greedy decode (\code{\link{yq_decoder}}) -> detokenize. Reuses the
#' package's pure-R tokenizer, suppress lists, and prompt builder. Single
#' 30 s window, no fallback/seek.
#'
#' @param audio_file Path to an audio file.
#' @param path Path to model.safetensors.
#' @param config A \code{\link{whisper_config}} list.
#' @param model Model name (for the tokenizer / special tokens).
#' @param language Language code for the prompt.
#' @param max_length Max tokens to generate.
#'
#' @return The transcribed text.
#'
#' @export
yq_transcribe <- function(audio_file, path, config, model = "tiny",
                          language = "en", max_length = 224L) {
  tok <- whisper_tokenizer(model)
  special <- whisper_special_tokens(model)
  enc_w <- yq_encoder_load_weights(path, config)
  dec_w <- yq_decoder_load_weights(path, config)
  mel <- yq_audio_to_mel(audio_file, n_mels = config$n_mels)
  # jit-compile the static-shape encoder (mel -> encoder output)
  enc_fn <- anvl::jit(function(m) yq_encoder(m, enc_w))
  xa <- enc_fn(mel)
  prompt <- get_initial_tokens(language = language, task = "transcribe",
    model = model, timestamps = FALSE)
  gen <- yq_greedy_decode(xa, dec_w, tok, special$eot, prompt, max_length)
  trimws(tok$decode(gen))
}
