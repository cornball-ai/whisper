#' Whisper Model
#'
#' Full Whisper model combining encoder and decoder.

#' Whisper Model Module
#'
#' @param config Model configuration
whisper_model <- torch::nn_module(
  "WhisperModel",

  initialize = function(config) {
    self$config <- config

    # Create encoder and decoder
    self$encoder <- create_encoder(config)
    self$decoder <- create_decoder(config)
  },

  forward = function(
    mel,
    tokens,
    kv_cache = NULL
  ) {
    # mel: (batch, n_mels, n_frames) mel spectrogram
    # tokens: (batch, seq_len) token ids

    # Encode audio
    encoder_output <- self$encoder(mel)

    # Decode with cross-attention
    decoder_result <- self$decoder(tokens, encoder_output, kv_cache = kv_cache)

    # Get logits
    logits <- self$decoder$get_logits(decoder_result$hidden_states)

    list(
      logits = logits,
      encoder_output = encoder_output,
      kv_cache = decoder_result$kv_cache
    )
  },

  encode = function(mel) {
    # Just encode audio (for caching encoder output)
    self$encoder(mel)
  },

  decode = function(
    tokens,
    encoder_output,
    kv_cache = NULL
  ) {
    # Just decode (with pre-computed encoder output)
    decoder_result <- self$decoder(tokens, encoder_output, kv_cache = kv_cache)
    logits <- self$decoder$get_logits(decoder_result$hidden_states)

    list(
      logits = logits,
      kv_cache = decoder_result$kv_cache
    )
  }
)

#' Load Whisper Model
#'
#' Load a Whisper model with weights from HuggingFace.
#'
#' @param model Model name: "tiny", "base", "small", "medium", "large-v3"
#' @param device Device to load model on ("auto", "cpu", "cuda")
#' @param dtype Data type ("auto", "float16", "float32")
#' @param download If TRUE, download model if not present
#' @param verbose Print loading messages
#' @return WhisperModel module
#' @export
load_whisper_model <- function(
  model = "tiny",
  device = "auto",
  dtype = "auto",
  download = TRUE,
  verbose = TRUE
) {
  # Parse device and dtype

  device <- parse_device(device)
  dtype <- parse_dtype(dtype, device)

  # Download model if needed
  if (download && !model_exists(model)) {
    download_whisper_model(model)
  }

  # Get config and create model
  config <- whisper_config(model)
  whisper <- whisper_model(config)

  # Load weights
  weights_path <- get_weights_path(model)
  load_whisper_weights(whisper, weights_path, verbose = verbose)

  # Move to device and dtype
  whisper$to(device = device, dtype = dtype)

  # Set eval mode
  whisper$eval()

  whisper
}

#' Load Weights from Safetensors
#'
#' @param model WhisperModel module
#' @param weights_path Path to safetensors file
#' @param verbose Print loading messages
load_whisper_weights <- function(
  model,
  weights_path,
  verbose = TRUE
) {
  if (!requireNamespace("safetensors", quietly = TRUE)) {
    stop("safetensors package required. Install with: install.packages('safetensors')")
  }

  if (verbose) message("Loading weights from: ", weights_path)

  # Load safetensors
  weights <- safetensors::safe_load_file(weights_path, framework = "torch")

  # Map HuggingFace weight names to our names
  torch::with_no_grad({
      load_encoder_weights(model$encoder, weights)
      load_decoder_weights(model$decoder, weights)
    })

  invisible(model)
}

#' Load Encoder Weights
#'
#' @param encoder WhisperEncoder module
#' @param weights Named list of tensors
load_encoder_weights <- function(
  encoder,
  weights
) {
  # Conv layers
  copy_if_exists(encoder$conv1$weight, weights, "encoder.conv1.weight")
  copy_if_exists(encoder$conv1$bias, weights, "encoder.conv1.bias")
  copy_if_exists(encoder$conv2$weight, weights, "encoder.conv2.weight")
  copy_if_exists(encoder$conv2$bias, weights, "encoder.conv2.bias")

  # Positional embedding (HF uses embed_positions.weight, may have model. prefix)
  pos_name <- "encoder.embed_positions.weight"
  if (!pos_name %in% names(weights)) {
    pos_name <- paste0("model.", pos_name)
  }
  if (pos_name %in% names(weights)) {
    pos_weight <- weights[[pos_name]]
    # Only copy up to our max length
    n_ctx <- min(encoder$n_ctx, pos_weight$size(1))
    encoder$positional_embedding[1:n_ctx,]$copy_(pos_weight[1:n_ctx,])
  }

  # Transformer layers
  for (i in seq_along(encoder$blocks)) {
    layer_prefix <- paste0("encoder.layers.", i - 1, ".")
    layer <- encoder$blocks[[i]]

    # Self-attention layer norm
    copy_if_exists(layer$attn_ln$weight, weights, paste0(layer_prefix, "self_attn_layer_norm.weight"))
    copy_if_exists(layer$attn_ln$bias, weights, paste0(layer_prefix, "self_attn_layer_norm.bias"))

    # Self-attention
    copy_if_exists(layer$attn$query$weight, weights, paste0(layer_prefix, "self_attn.q_proj.weight"))
    copy_if_exists(layer$attn$query$bias, weights, paste0(layer_prefix, "self_attn.q_proj.bias"))
    copy_if_exists(layer$attn$key$weight, weights, paste0(layer_prefix, "self_attn.k_proj.weight"))
    copy_if_exists(layer$attn$value$weight, weights, paste0(layer_prefix, "self_attn.v_proj.weight"))
    copy_if_exists(layer$attn$value$bias, weights, paste0(layer_prefix, "self_attn.v_proj.bias"))
    copy_if_exists(layer$attn$out$weight, weights, paste0(layer_prefix, "self_attn.out_proj.weight"))
    copy_if_exists(layer$attn$out$bias, weights, paste0(layer_prefix, "self_attn.out_proj.bias"))

    # FFN layer norm
    copy_if_exists(layer$mlp_ln$weight, weights, paste0(layer_prefix, "final_layer_norm.weight"))
    copy_if_exists(layer$mlp_ln$bias, weights, paste0(layer_prefix, "final_layer_norm.bias"))

    # FFN
    copy_if_exists(layer$mlp[[1]]$weight, weights, paste0(layer_prefix, "fc1.weight"))
    copy_if_exists(layer$mlp[[1]]$bias, weights, paste0(layer_prefix, "fc1.bias"))
    copy_if_exists(layer$mlp[[3]]$weight, weights, paste0(layer_prefix, "fc2.weight"))
    copy_if_exists(layer$mlp[[3]]$bias, weights, paste0(layer_prefix, "fc2.bias"))
  }

  # Final layer norm
  copy_if_exists(encoder$ln_post$weight, weights, "encoder.layer_norm.weight")
  copy_if_exists(encoder$ln_post$bias, weights, "encoder.layer_norm.bias")
}

#' Load Decoder Weights
#'
#' @param decoder WhisperDecoder module
#' @param weights Named list of tensors
load_decoder_weights <- function(
  decoder,
  weights
) {
  # Token embedding
  copy_if_exists(decoder$token_embedding$weight, weights, "decoder.embed_tokens.weight")

  # Positional embedding
  copy_if_exists(decoder$positional_embedding$weight, weights, "decoder.embed_positions.weight")

  # Transformer layers
  for (i in seq_along(decoder$blocks)) {
    layer_prefix <- paste0("decoder.layers.", i - 1, ".")
    layer <- decoder$blocks[[i]]

    # Self-attention layer norm
    copy_if_exists(layer$attn_ln$weight, weights, paste0(layer_prefix, "self_attn_layer_norm.weight"))
    copy_if_exists(layer$attn_ln$bias, weights, paste0(layer_prefix, "self_attn_layer_norm.bias"))

    # Self-attention
    copy_if_exists(layer$attn$query$weight, weights, paste0(layer_prefix, "self_attn.q_proj.weight"))
    copy_if_exists(layer$attn$query$bias, weights, paste0(layer_prefix, "self_attn.q_proj.bias"))
    copy_if_exists(layer$attn$key$weight, weights, paste0(layer_prefix, "self_attn.k_proj.weight"))
    copy_if_exists(layer$attn$value$weight, weights, paste0(layer_prefix, "self_attn.v_proj.weight"))
    copy_if_exists(layer$attn$value$bias, weights, paste0(layer_prefix, "self_attn.v_proj.bias"))
    copy_if_exists(layer$attn$out$weight, weights, paste0(layer_prefix, "self_attn.out_proj.weight"))
    copy_if_exists(layer$attn$out$bias, weights, paste0(layer_prefix, "self_attn.out_proj.bias"))

    # Cross-attention layer norm
    copy_if_exists(layer$cross_attn_ln$weight, weights, paste0(layer_prefix, "encoder_attn_layer_norm.weight"))
    copy_if_exists(layer$cross_attn_ln$bias, weights, paste0(layer_prefix, "encoder_attn_layer_norm.bias"))

    # Cross-attention
    copy_if_exists(layer$cross_attn$query$weight, weights, paste0(layer_prefix, "encoder_attn.q_proj.weight"))
    copy_if_exists(layer$cross_attn$query$bias, weights, paste0(layer_prefix, "encoder_attn.q_proj.bias"))
    copy_if_exists(layer$cross_attn$key$weight, weights, paste0(layer_prefix, "encoder_attn.k_proj.weight"))
    copy_if_exists(layer$cross_attn$value$weight, weights, paste0(layer_prefix, "encoder_attn.v_proj.weight"))
    copy_if_exists(layer$cross_attn$value$bias, weights, paste0(layer_prefix, "encoder_attn.v_proj.bias"))
    copy_if_exists(layer$cross_attn$out$weight, weights, paste0(layer_prefix, "encoder_attn.out_proj.weight"))
    copy_if_exists(layer$cross_attn$out$bias, weights, paste0(layer_prefix, "encoder_attn.out_proj.bias"))

    # FFN layer norm
    copy_if_exists(layer$mlp_ln$weight, weights, paste0(layer_prefix, "final_layer_norm.weight"))
    copy_if_exists(layer$mlp_ln$bias, weights, paste0(layer_prefix, "final_layer_norm.bias"))

    # FFN
    copy_if_exists(layer$mlp[[1]]$weight, weights, paste0(layer_prefix, "fc1.weight"))
    copy_if_exists(layer$mlp[[1]]$bias, weights, paste0(layer_prefix, "fc1.bias"))
    copy_if_exists(layer$mlp[[3]]$weight, weights, paste0(layer_prefix, "fc2.weight"))
    copy_if_exists(layer$mlp[[3]]$bias, weights, paste0(layer_prefix, "fc2.bias"))
  }

  # Final layer norm
  copy_if_exists(decoder$ln$weight, weights, "decoder.layer_norm.weight")
  copy_if_exists(decoder$ln$bias, weights, "decoder.layer_norm.bias")
}

#' Copy Weight if Exists
#'
#' @param param Target parameter
#' @param weights Weight dictionary
#' @param name Weight name
copy_if_exists <- function(
  param,
  weights,
  name
) {
  if (name %in% names(weights)) {
    param$copy_(weights[[name]])
  } else {
    # Try with model. prefix (HuggingFace format)
    alt_name <- paste0("model.", name)
    if (alt_name %in% names(weights)) {
      param$copy_(weights[[alt_name]])
    }
  }
}

