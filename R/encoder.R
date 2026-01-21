#' Whisper Encoder
#'
#' Transformer encoder for processing mel spectrograms.

#' Multi-Head Self-Attention
#'
#' @param n_state Hidden dimension
#' @param n_head Number of attention heads
whisper_attention <- torch::nn_module(
  "WhisperAttention",

  initialize = function(
    n_state,
    n_head
  ) {
    self$n_head <- n_head
    self$n_state <- n_state
    self$head_dim <- n_state %/% n_head

    # Combined QKV projection
    self$query <- torch::nn_linear(n_state, n_state)
    self$key <- torch::nn_linear(n_state, n_state, bias = FALSE)
    self$value <- torch::nn_linear(n_state, n_state)

    # Output projection
    self$out <- torch::nn_linear(n_state, n_state)

    self$scale <- self$head_dim ^ (- 0.5)
  },

  forward = function(
    x,
    xa = NULL,
    mask = NULL,
    kv_cache = NULL
  ) {
    # x: (batch, seq_len, n_state)
    # xa: optional cross-attention input (batch, src_len, n_state)

    batch_size <- x$size(1)
    seq_len <- x$size(2)

    # Query from x
    q <- self$query(x)
    q <- self$reshape_for_attention(q, batch_size)

    # Key/Value handling differs for self-attention vs cross-attention
    if (!is.null(xa)) {
      # Cross-attention: K,V come from encoder output
      if (!is.null(kv_cache)) {
        # Reuse cached encoder K,V (encoder output doesn't change)
        k <- kv_cache$k
        v <- kv_cache$v
      } else {
        # First time: compute K,V from encoder output
        k <- self$key(xa)
        v <- self$value(xa)
        k <- self$reshape_for_attention(k, batch_size)
        v <- self$reshape_for_attention(v, batch_size)
      }
    } else {
      # Self-attention: K,V come from decoder input
      k <- self$key(x)
      v <- self$value(x)
      k <- self$reshape_for_attention(k, batch_size)
      v <- self$reshape_for_attention(v, batch_size)

      # Concatenate with cache for autoregressive decoding
      if (!is.null(kv_cache)) {
        k <- torch::torch_cat(list(kv_cache$k, k), dim = 3L)
        v <- torch::torch_cat(list(kv_cache$v, v), dim = 3L)
      }
    }

    # Scaled dot-product attention
    # (batch, n_head, seq_len, head_dim) @ (batch, n_head, head_dim, src_len)
    # -> (batch, n_head, seq_len, src_len)
    scores <- torch::torch_matmul(q, k$transpose(3L, 4L))$mul(self$scale)

    # Apply mask if provided
    if (!is.null(mask)) {
      scores <- scores + mask
    }

    # Softmax
    attn_weights <- torch::nnf_softmax(scores, dim = - 1L)

    # Apply attention to values
    # (batch, n_head, seq_len, src_len) @ (batch, n_head, src_len, head_dim)
    # -> (batch, n_head, seq_len, head_dim)
    attn_output <- torch::torch_matmul(attn_weights, v)

    # Reshape back
    # (batch, n_head, seq_len, head_dim) -> (batch, seq_len, n_state)
    attn_output <- attn_output$transpose(2L, 3L)$contiguous()
    attn_output <- attn_output$view(c(batch_size, seq_len, self$n_state))

    # Output projection
    output <- self$out(attn_output)

    # Return output and new KV cache (reshaped k, v for efficient caching)
    list(output = output, kv_cache = list(k = k, v = v))
  },

  reshape_for_attention = function(
    x,
    batch_size
  ) {
    # (batch, seq_len, n_state) -> (batch, n_head, seq_len, head_dim)
    seq_len <- x$size(2)
    x$view(c(batch_size, seq_len, self$n_head, self$head_dim))$transpose(2L, 3L)
  }
)

#' Encoder Layer
#'
#' Pre-norm transformer encoder layer.
#'
#' @param n_state Hidden dimension
#' @param n_head Number of attention heads
whisper_encoder_layer <- torch::nn_module(
  "WhisperEncoderLayer",

  initialize = function(
    n_state,
    n_head
  ) {
    self$attn_ln <- torch::nn_layer_norm(n_state)
    self$attn <- whisper_attention(n_state, n_head)

    self$mlp_ln <- torch::nn_layer_norm(n_state)
    self$mlp <- torch::nn_sequential(
      torch::nn_linear(n_state, n_state * 4L),
      torch::nn_gelu(),
      torch::nn_linear(n_state * 4L, n_state)
    )
  },

  forward = function(x) {
    # Self-attention with residual
    attn_result <- self$attn(self$attn_ln(x))
    x <- x + attn_result$output

    # FFN with residual
    x <- x + self$mlp(self$mlp_ln(x))

    x
  }
)

#' Audio Encoder
#'
#' Full Whisper encoder: Conv stem + positional encoding + transformer layers.
#'
#' @param n_mels Number of mel spectrogram bins
#' @param n_ctx Maximum context length (1500 for 30s audio)
#' @param n_state Hidden dimension
#' @param n_head Number of attention heads
#' @param n_layer Number of transformer layers
whisper_encoder <- torch::nn_module(
  "WhisperEncoder",

  initialize = function(
    n_mels,
    n_ctx,
    n_state,
    n_head,
    n_layer
  ) {
    self$n_mels <- n_mels
    self$n_ctx <- n_ctx
    self$n_state <- n_state

    # Convolutional stem
    # Conv1d: in_channels, out_channels, kernel_size
    self$conv1 <- torch::nn_conv1d(n_mels, n_state, kernel_size = 3L, padding = 1L)
    self$conv2 <- torch::nn_conv1d(n_state, n_state, kernel_size = 3L, stride = 2L, padding = 1L)

    # Positional encoding (sinusoidal, registered as buffer)
    self$register_buffer("positional_embedding", self$create_sinusoidal_pe(n_ctx, n_state))

    # Transformer layers
    self$blocks <- torch::nn_module_list()
    for (i in seq_len(n_layer)) {
      self$blocks$append(whisper_encoder_layer(n_state, n_head))
    }

    # Final layer norm
    self$ln_post <- torch::nn_layer_norm(n_state)
  },

  create_sinusoidal_pe = function(
    max_len,
    dim
  ) {
    # Create sinusoidal positional embeddings
    pe <- torch::torch_zeros(max_len, dim)

    position <- torch::torch_arange(0, max_len - 1, dtype = torch::torch_float())$unsqueeze(2L)
    div_term <- torch::torch_exp(
      torch::torch_arange(0, dim - 1, 2, dtype = torch::torch_float())$mul(- log(10000.0) / dim)
    )

    # Sin for even indices, cos for odd
    pe[, seq(1, dim, 2)] <- torch::torch_sin(position * div_term)
    pe[, seq(2, dim, 2)] <- torch::torch_cos(position * div_term)

    pe
  },

  forward = function(x) {
    # x: (batch, n_mels, n_frames) mel spectrogram

    # Conv stem with GELU
    x <- torch::nnf_gelu(self$conv1(x))
    x <- torch::nnf_gelu(self$conv2(x))

    # (batch, n_state, n_frames/2) -> (batch, n_frames/2, n_state)
    x <- x$permute(c(1L, 3L, 2L))

    # Get sequence length after convolutions
    seq_len <- x$size(2)

    # Truncate if longer than max context (can happen due to STFT edge effects)
    if (seq_len > self$n_ctx) {
      x <- x[, 1:self$n_ctx,]
      seq_len <- self$n_ctx
    }

    # Add positional encoding
    # Slice positional embedding to match sequence length
    pos_emb <- self$positional_embedding[1:seq_len,]
    x <- x + pos_emb$unsqueeze(1L)

    # Transformer layers
    for (i in seq_along(self$blocks)) {
      x <- self$blocks[[i]](x)
    }

    # Final layer norm
    x <- self$ln_post(x)

    x
  }
)

#' Create Encoder from Config
#'
#' @param config Model configuration from whisper_config()
#' @return WhisperEncoder module
create_encoder <- function(config) {
  whisper_encoder(
    n_mels = config$n_mels,
    n_ctx = config$n_audio_ctx,
    n_state = config$n_audio_state,
    n_head = config$n_audio_head,
    n_layer = config$n_audio_layer
  )
}

