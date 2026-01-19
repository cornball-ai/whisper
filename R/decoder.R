#' Whisper Decoder
#'
#' Transformer decoder with cross-attention to encoder outputs.

#' Decoder Layer
#'
#' Pre-norm transformer decoder layer with self-attention and cross-attention.
#'
#' @param n_state Hidden dimension
#' @param n_head Number of attention heads
whisper_decoder_layer <- torch::nn_module(
  "WhisperDecoderLayer",

  initialize = function(
    n_state,
    n_head
  ) {
    # Self-attention
    self$attn_ln <- torch::nn_layer_norm(n_state)
    self$attn <- whisper_attention(n_state, n_head)

    # Cross-attention
    self$cross_attn_ln <- torch::nn_layer_norm(n_state)
    self$cross_attn <- whisper_attention(n_state, n_head)

    # FFN
    self$mlp_ln <- torch::nn_layer_norm(n_state)
    self$mlp <- torch::nn_sequential(
      torch::nn_linear(n_state, n_state * 4L),
      torch::nn_gelu(),
      torch::nn_linear(n_state * 4L, n_state)
    )
  },

  forward = function(
    x,
    xa,
    mask = NULL,
    kv_cache = NULL
  ) {
    # x: decoder input (batch, seq_len, n_state)
    # xa: encoder output (batch, src_len, n_state)
    # mask: causal attention mask
    # kv_cache: cached key-value pairs for incremental decoding

    # Extract caches if provided
    self_kv_cache <- NULL
    cross_kv_cache <- NULL
    if (!is.null(kv_cache)) {
      self_kv_cache <- kv_cache$self
      cross_kv_cache <- kv_cache$cross
    }

    # Self-attention with causal mask
    attn_result <- self$attn(self$attn_ln(x), mask = mask, kv_cache = self_kv_cache)
    x <- x + attn_result$output
    new_self_kv <- attn_result$kv_cache

    # Cross-attention to encoder output
    cross_result <- self$cross_attn(self$cross_attn_ln(x), xa = xa, kv_cache = cross_kv_cache)
    x <- x + cross_result$output
    new_cross_kv <- cross_result$kv_cache

    # FFN
    x <- x + self$mlp(self$mlp_ln(x))

    # Return output and updated caches
    list(
      output = x,
      kv_cache = list(self = new_self_kv, cross = new_cross_kv)
    )
  }
)

#' Text Decoder
#'
#' Full Whisper decoder: token embedding + positional embedding + transformer layers.
#'
#' @param n_vocab Vocabulary size
#' @param n_ctx Maximum context length
#' @param n_state Hidden dimension
#' @param n_head Number of attention heads
#' @param n_layer Number of transformer layers
whisper_decoder <- torch::nn_module(
  "WhisperDecoder",

  initialize = function(
    n_vocab,
    n_ctx,
    n_state,
    n_head,
    n_layer
  ) {
    self$n_vocab <- n_vocab
    self$n_ctx <- n_ctx
    self$n_state <- n_state
    self$n_layer <- n_layer

    # Token embedding
    self$token_embedding <- torch::nn_embedding(n_vocab, n_state)

    # Learned positional embedding (unlike encoder's sinusoidal)
    self$positional_embedding <- torch::nn_embedding(n_ctx, n_state)

    # Transformer layers
    self$blocks <- torch::nn_module_list()
    for (i in seq_len(n_layer)) {
      self$blocks$append(whisper_decoder_layer(n_state, n_head))
    }

    # Final layer norm
    self$ln <- torch::nn_layer_norm(n_state)

    # Register causal mask buffer
    self$register_buffer("mask", self$create_causal_mask(n_ctx))
  },

  create_causal_mask = function(n_ctx) {
    # Create causal (lower triangular) mask
    # 0 where attention is allowed, -inf where blocked
    mask <- torch::torch_full(c(n_ctx, n_ctx), - Inf)
    mask <- torch::torch_triu(mask, diagonal = 1L)
    mask
  },

  forward = function(
    x,
    xa,
    kv_cache = NULL
  ) {
    # x: token ids (batch, seq_len)
    # xa: encoder output (batch, src_len, n_state)
    # kv_cache: list of cached key-value pairs per layer

    batch_size <- x$size(1)
    seq_len <- x$size(2)

    # Determine position offset for incremental decoding
    offset <- 0L
    if (!is.null(kv_cache) && !is.null(kv_cache[[1]]$self$k)) {
      offset <- kv_cache[[1]]$self$k$size(3)
    }

    # Token embedding (R torch uses 1-based indexing, so add 1 to token IDs)
    x <- self$token_embedding(x + 1L)

    # Position indices (R torch uses 1-based indexing)
    positions <- torch::torch_arange(offset + 1L, offset + seq_len,
      dtype = torch::torch_long(),
      device = x$device)

    # Add positional embedding
    pos_emb <- self$positional_embedding(positions)
    x <- x + pos_emb$unsqueeze(1L)

    # Get causal mask for current sequence
    # For incremental decoding with cache, we only need mask for new positions
    if (is.null(kv_cache)) {
      mask <- self$mask[1:seq_len, 1:seq_len]
    } else {
      # During incremental decoding, no mask needed (single token at a time)
      mask <- NULL
    }

    # Initialize or use provided KV cache
    if (is.null(kv_cache)) {
      kv_cache <- vector("list", self$n_layer)
    }

    new_kv_cache <- vector("list", self$n_layer)

    # Transformer layers
    for (i in seq_len(self$n_layer)) {
      if (i <= length(kv_cache)) {
        layer_cache <- kv_cache[[i]]
      } else {
        layer_cache <- NULL
      }
      result <- self$blocks[[i]](x, xa, mask = mask, kv_cache = layer_cache)
      x <- result$output
      new_kv_cache[[i]] <- result$kv_cache
    }

    # Final layer norm
    x <- self$ln(x)

    # Return hidden states and updated KV cache
    list(hidden_states = x, kv_cache = new_kv_cache)
  },

  get_logits = function(hidden_states) {
    # Project to vocabulary logits using transposed token embedding
    # hidden_states: (batch, seq_len, n_state)
    # token_embedding.weight: (n_vocab, n_state)
    # output: (batch, seq_len, n_vocab)

    torch::torch_matmul(hidden_states, self$token_embedding$weight$t())
  }
)

#' Create Decoder from Config
#'
#' @param config Model configuration from whisper_config()
#' @return WhisperDecoder module
create_decoder <- function(config) {
  whisper_decoder(
    n_vocab = config$n_vocab,
    n_ctx = config$n_text_ctx,
    n_state = config$n_text_state,
    n_head = config$n_text_head,
    n_layer = config$n_text_layer
  )
}

