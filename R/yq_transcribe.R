# anvl/yunque end-to-end transcription. Greedy decode without a KV cache:
# recompute the decoder over the growing prompt each step (O(S^2); correct
# and simple -- the KV cache is the optimization). Suppress masks and the EOT
# stop match the torch greedy path.

yq_greedy_decode <- function(xa, dec_w, tok, eot, prompt, max_length = 224L) {
  tokens <- matrix(as.integer(prompt), nrow = 1L)
  suppress <- tok$suppress_tokens
  blank <- tok$blank_tokens
  generated <- integer(0)
  for (step in seq_len(max_length)) {
    logits <- as.array(yq_decoder(tokens, xa, dec_w))
    nl <- logits[1L, ncol(tokens), ] # last-position logits, length n_vocab
    nl[suppress + 1L] <- -Inf # 0-based ids -> R 1-based
    if (step == 1L) {
      nl[blank + 1L] <- -Inf # SuppressBlank: first step only
    }
    nxt <- which.max(nl) - 1L # 0-based token id
    if (nxt == eot) {
      break
    }
    generated <- c(generated, nxt)
    tokens <- cbind(tokens, nxt)
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
  xa <- yq_encoder(mel, enc_w)
  prompt <- get_initial_tokens(language = language, task = "transcribe",
    model = model, timestamps = FALSE)
  gen <- yq_greedy_decode(xa, dec_w, tok, special$eot, prompt, max_length)
  trimws(tok$decode(gen))
}
