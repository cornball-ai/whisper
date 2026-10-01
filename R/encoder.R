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
  },

  forward = function(
    x,
    xa = NULL,
    mask = NULL,
    kv_cache = NULL,
    need_weights = FALSE
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

    attn_weights <- NULL

    if (need_weights) {
      # Manual attention to capture weights for DTW alignment
      # q: (batch, n_head, seq_len, head_dim)
      # k: (batch, n_head, src_len, head_dim)
      scale <- sqrt(self$head_dim)
      attn_scores <- torch::torch_matmul(q, k$transpose(3L, 4L)) / scale
      if (!is.null(mask)) {
        attn_scores <- attn_scores + mask
      }
      attn_weights <- torch::nnf_softmax(attn_scores, dim = -1L)
      attn_output <- torch::torch_matmul(attn_weights, v)
    } else {
      # Scaled dot-product attention (dispatches to FlashAttention on GPU).
      attn_output <- torch::torch_scaled_dot_product_attention(
        q, k, v, is_causal = !is.null(mask))
    }

    # Reshape back: (batch, n_head, seq_len, head_dim) -> (batch, seq_len, n_state)
    attn_output <- attn_output$transpose(2L, 3L)$contiguous()
    attn_output <- attn_output$view(c(batch_size, seq_len, self$n_state))

    # Output projection
    output <- self$out(attn_output)

    # Return output, KV cache, and optionally attention weights
    list(output = output, kv_cache = list(k = k, v = v),
      attn_weights = attn_weights)
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
    fn <- make_encoder_layer_fn(self$attn$n_head, self$attn$head_dim)
    do.call(fn, c(list(x), self$weights()))
  },

  # The layer's tensors in encoder_layer_fn()'s argument order.
  weights = function() {
    list(self$attn_ln$weight, self$attn_ln$bias,
      self$attn$query$weight, self$attn$query$bias,
      self$attn$key$weight,
      self$attn$value$weight, self$attn$value$bias,
      self$attn$out$weight, self$attn$out$bias,
      self$mlp_ln$weight, self$mlp_ln$bias,
      self$mlp[[1]]$weight, self$mlp[[1]]$bias,
      self$mlp[[3]]$weight, self$mlp[[3]]$bias)
  }
)

#' One Encoder Layer as a Function of Tensors
#'
#' Pre-norm self-attention (no mask) then a GELU MLP, each with a residual:
#' the computation of whisper_encoder_layer, written so that every input is
#' a tensor. One traced copy can then serve every layer, with the layer's
#' weights passed in (see encoder_layer_fn_for()). The sequence length is
#' never read, so a trace accepts any length.
#'
#' @param n_head,head_dim Attention geometry, fixed by the closure.
#' @return A function of \code{(x, <15 layer weights>)}.
#' @noRd
make_encoder_layer_fn <- function(n_head, head_dim) {
  n_state <- n_head * head_dim
  function(x, attn_ln_w, attn_ln_b, q_w, q_b, k_w, v_w, v_b, out_w, out_b,
           mlp_ln_w, mlp_ln_b, fc1_w, fc1_b, fc2_w, fc2_b) {
    b <- x$size(1)
    split <- function(t) {
      t$view(c(b, -1L, n_head, head_dim))$transpose(2L, 3L)
    }
    h <- torch::nnf_layer_norm(x, n_state, attn_ln_w, attn_ln_b)
    q <- split(torch::nnf_linear(h, q_w, q_b))
    k <- split(torch::nnf_linear(h, k_w))
    v <- split(torch::nnf_linear(h, v_w, v_b))
    a <- torch::torch_scaled_dot_product_attention(q, k, v)
    a <- a$transpose(2L, 3L)$reshape(c(b, -1L, n_state))
    x <- x + torch::nnf_linear(a, out_w, out_b)
    h <- torch::nnf_layer_norm(x, n_state, mlp_ln_w, mlp_ln_b)
    h <- torch::nnf_gelu(torch::nnf_linear(h, fc1_w, fc1_b))
    x + torch::nnf_linear(h, fc2_w, fc2_b)
  }
}

# Traced encoder layer functions, keyed by architecture, device, dtype and
# batch size (the batch size is read from the input, so a trace fixes it).
.whisper_jit_encoder_cache <- new.env(parent = emptyenv())

#' The Encoder Layer Function to Run on This Input
#'
#' On CUDA, the layer function traced to TorchScript. Run op by op from R,
#' every intermediate of every layer stays allocated until R's garbage
#' collector runs; for a 30 s window that more than doubles the encoder's
#' peak device memory (whisper-small fp32: 2.36 GiB against a 0.99 GiB
#' working set). Inside a traced graph intermediates are freed as they are
#' consumed. Tracing one layer with the weights as inputs, rather than the
#' whole encoder, keeps TorchScript's compile and warm-up short, and the
#' trace holds no weights, so resident swaps cannot leave it stale. The CPU
#' gains nothing from it and runs the plain function, as does
#' \code{options(whisper.jit = FALSE)}.
#'
#' @param block Any whisper_encoder_layer of the model (for its geometry
#'   and example weights).
#' @param x The layer input.
#' @return A function of \code{(x, <15 layer weights>)}.
#' @noRd
encoder_layer_fn_for <- function(block, x) {
  n_head <- block$attn$n_head
  head_dim <- block$attn$head_dim
  plain <- make_encoder_layer_fn(n_head, head_dim)
  if (x$device$type != "cuda" || !isTRUE(getOption("whisper.jit", TRUE))) {
    return(plain)
  }
  key <- paste(n_head, head_dim, x$device$index, x$dtype$.type(), x$size(1))
  if (is.null(.whisper_jit_encoder_cache[[key]])) {
    .whisper_jit_encoder_cache[[key]] <- trace_encoder_layer(plain, block, x)
  }
  .whisper_jit_encoder_cache[[key]]
}

# Traces the layer function and runs it at three lengths. TorchScript's
# profiling executor specializes on static shapes twice, then compiles a
# dynamic-shape graph; after these runs any length is served by that graph,
# so no real window pays for compilation.
trace_encoder_layer <- function(fn, block, x) {
  weights <- block$weights()
  example <- function(l) {
    c(list(torch::torch_zeros(x$size(1), l, x$size(3), dtype = x$dtype,
      device = x$device)), weights)
  }
  torch::with_no_grad({
    traced <- do.call(torch::jit_trace, c(list(fn), example(16L)))
    for (l in c(16L, 16L, 17L, 17L, 18L, 18L, 18L)) {
      do.call(traced, example(l))
    }
  })
  traced
}

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

    # Conv stem with GELU; off cuDNN where its fp16 conv is broken
    if (.stem_needs_unfold(x)) {
      x <- torch::nnf_gelu(conv1d_unfold(x, self$conv1$weight,
        self$conv1$bias))
      x <- torch::nnf_gelu(conv1d_unfold(x, self$conv2$weight,
        self$conv2$bias, stride = 2L))
    } else {
      x <- torch::nnf_gelu(self$conv1(x))
      x <- torch::nnf_gelu(self$conv2(x))
    }

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

    # Transformer layers: one layer function, each block's weights
    fn <- encoder_layer_fn_for(self$blocks[[1]], x)
    for (i in seq_along(self$blocks)) {
      x <- do.call(fn, c(list(x), self$blocks[[i]]$weights()))
    }

    # Final layer norm
    x <- self$ln_post(x)

    # The layer outputs and stem temporaries are dead but held until R
    # collects; on a GPU a minor collection (a few ms) releases them now.
    if (x$device$type == "cuda") {
      invisible(gc(full = FALSE))
    }

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

