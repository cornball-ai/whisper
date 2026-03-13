#' Language Detection
#'
#' Detect the spoken language in an audio file using Whisper.

#' Detect Language
#'
#' Identify the spoken language in an audio file. Uses Whisper's decoder
#' to predict the most likely language token from the first 30 seconds
#' of audio.
#'
#' @param file Path to audio file (WAV, MP3, etc.)
#' @param model Model name: "tiny", "base", "small", "medium", "large-v3"
#' @param device Device: "auto", "cpu", "cuda"
#' @param dtype Data type: "auto", "float16", "float32"
#' @param top_k Number of top language probabilities to return (default: 5)
#' @param download If TRUE and model not present, prompt to download.
#' @param verbose Print loading messages.
#' @return List with \code{language} (two-letter code) and
#'   \code{probabilities} (named numeric vector of top-k language probs).
#' @export
#' @examples
#' \donttest{
#' if (model_exists("tiny")) {
#'   audio_file <- system.file("audio", "jfk.mp3", package = "whisper")
#'   result <- detect_language(audio_file)
#'   result$language
#'   result$probabilities
#' }
#' }
detect_language <- function(
  file,
  model = "tiny",
  device = "auto",
  dtype = "auto",
  top_k = 5L,
  download = TRUE,
  verbose = TRUE
) {
  pipe <- whisper_pipeline(model, device = device, dtype = dtype,
    download = download, verbose = verbose)

  detect_language_from_pipeline(pipe, file, top_k = top_k)
}

#' Detect Language from Pipeline
#'
#' Internal function that runs language detection using a pre-loaded pipeline.
#'
#' @param pipe A whisper_pipeline object
#' @param file Path to audio file, or numeric vector of audio samples
#' @param top_k Number of top probabilities to return
#' @return List with language code and probabilities
detect_language_from_pipeline <- function(pipe, file, top_k = 5L) {
  config <- pipe$config
  model <- pipe$model
  device <- pipe$device
  dtype <- pipe$dtype

  # Compute mel spectrogram from first 30s
  mel <- audio_to_mel(file, n_mels = config$n_mels, device = device,
    dtype = dtype)

  detect_language_from_mel(model, mel, config, device)
}

#' Detect Language from Mel Spectrogram
#'
#' Core detection logic. Feed SOT token to decoder, read language logits.
#'
#' @param model WhisperModel
#' @param mel Mel spectrogram tensor
#' @param config Model config
#' @param device torch device
#' @param top_k Number of top probabilities to return
#' @return List with language code and probabilities
detect_language_from_mel <- function(model, mel, config, device, top_k = 5L) {
  special <- whisper_special_tokens(config$model_name)
  langs <- whisper_language_table()
  n_langs <- length(langs)

  # Language token IDs: 50259 to 50259 + n_langs - 1
  lang_start <- 50259L
  lang_end <- lang_start + n_langs - 1L

  torch::with_no_grad({
    # Encode audio
    encoder_output <- model$encode(mel)

    # Feed just the SOT token to the decoder
    sot <- torch::torch_tensor(matrix(special$sot, nrow = 1L),
      dtype = torch::torch_long(), device = device)

    result <- model$decode(sot, encoder_output)
    # logits shape: (1, 1, n_vocab)
    logits <- result$logits[1, 1, ]

    # Extract language token logits (R is 1-indexed, token IDs are 0-indexed in vocab)
    # Token ID 50259 is at position 50260 in 1-indexed logits
    lang_logits <- logits[(lang_start + 1L):(lang_end + 1L)]

    # Softmax over language logits only
    probs <- torch::nnf_softmax(lang_logits, dim = 1L)
    probs_r <- as.numeric(probs$cpu())
  })

  names(probs_r) <- names(langs)

  # Find top-k
  top_idx <- order(probs_r, decreasing = TRUE)[seq_len(min(top_k, n_langs))]
  top_probs <- probs_r[top_idx]

  list(
    language = names(probs_r)[top_idx[1]],
    probabilities = top_probs
  )
}
