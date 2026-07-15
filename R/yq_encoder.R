# anvl/yunque port of the Whisper audio encoder (conv stem + transformer).
# Torch-free; yq_ marks the anvl/XLA implementation vs the torch one.

# [B, S, n_state] -> [B, n_head, S, head_dim]
.yq_heads <- function(x, batch, n_head, head_dim) {
  s <- anvl::shape(x)
  anvl::nv_transpose(anvl::nv_reshape(x, c(batch, s[2L], n_head, head_dim)),
    c(1L, 3L, 2L, 4L))
}

#' Load Whisper encoder weights (anvl)
#'
#' Reads an HF Whisper safetensors file into anvl arrays for
#' \code{\link{yq_encoder}}. Linear weights are transposed to
#' \code{[in, out]} for \code{yunque::linear}; conv weights stay
#' \code{[C_out, C_in, kW]}; the positional table is sliced to
#' \code{n_audio_ctx}.
#'
#' @param path Path to model.safetensors.
#' @param config A \code{\link{whisper_config}} list.
#' @param prefix Key prefix (HF checkpoints use \code{"model.encoder."}).
#'
#' @return A named list of anvl arrays plus \code{config}.
#'
#' @export
yq_encoder_load_weights <- function(path, config, prefix = "model.encoder.") {
  st <- yunque::st_open(path)
  on.exit(yunque::st_close(st))
  has <- function(k) !is.null(st$header[[paste0(prefix, k)]])
  nv <- function(k, transpose = FALSE) {
    anvl::nv_array(yunque::st_read(st, paste0(prefix, k), transpose = transpose),
      dtype = "f32")
  }
  lin <- function(base) list(
    w = nv(paste0(base, ".weight"), transpose = TRUE),
    b = if (has(paste0(base, ".bias"))) nv(paste0(base, ".bias")) else NULL
  )
  ln <- function(base) list(w = nv(paste0(base, ".weight")),
    b = nv(paste0(base, ".bias")))

  layers <- lapply(seq_len(config$n_audio_layer) - 1L, function(i) {
    p <- paste0("layers.", i, ".")
    list(
      attn_ln = ln(paste0(p, "self_attn_layer_norm")),
      q = lin(paste0(p, "self_attn.q_proj")),
      k = lin(paste0(p, "self_attn.k_proj")), # no bias
      v = lin(paste0(p, "self_attn.v_proj")),
      out = lin(paste0(p, "self_attn.out_proj")),
      mlp_ln = ln(paste0(p, "final_layer_norm")), # HF name; pre-MLP norm
      fc1 = lin(paste0(p, "fc1")),
      fc2 = lin(paste0(p, "fc2"))
    )
  })
  pos <- yunque::st_read(st, paste0(prefix, "embed_positions.weight"))
  list(
    conv1 = list(w = nv("conv1.weight"), b = nv("conv1.bias")),
    conv2 = list(w = nv("conv2.weight"), b = nv("conv2.bias")),
    pos_emb = anvl::nv_array(pos[seq_len(config$n_audio_ctx), , drop = FALSE],
      dtype = "f32"),
    layers = layers,
    ln_post = ln("layer_norm"),
    config = config
  )
}

#' Whisper audio encoder forward (anvl)
#'
#' Torch-free port of \code{whisper_encoder}: conv stem (GELU, stride-2
#' downsample) + positional embedding + pre-norm transformer layers +
#' final LayerNorm. GELU is exact (erf); LayerNorm affine, eps 1e-5.
#'
#' @param mel AnvlArray \code{[1, n_mels, 3000]} (from
#'   \code{\link{yq_audio_to_mel}}).
#' @param w Weights from \code{\link{yq_encoder_load_weights}}.
#'
#' @return AnvlArray \code{[1, n_audio_ctx, n_audio_state]}.
#'
#' @export
yq_encoder <- function(mel, w) {
  cfg <- w$config
  nh <- cfg$n_audio_head
  hd <- cfg$n_audio_state %/% nh
  eps <- 1e-5

  x <- yunque::gelu(yunque::conv1d(mel, w$conv1$w, w$conv1$b, padding = 1L),
    "none")
  x <- yunque::gelu(
    yunque::conv1d(x, w$conv2$w, w$conv2$b, stride = 2L, padding = 1L), "none")
  x <- anvl::nv_transpose(x, c(1L, 3L, 2L)) # [B, T, n_state]
  s <- anvl::shape(x)
  batch <- s[1L]
  seq <- s[2L]
  ns <- s[3L]
  if (seq > cfg$n_audio_ctx) {
    x <- yunque::slice_seq(x, 1L, cfg$n_audio_ctx)
    seq <- cfg$n_audio_ctx
  }
  x <- x + anvl::nv_broadcast_to(anvl::nv_unsqueeze(w$pos_emb, 1L),
    anvl::shape(x))

  for (ly in w$layers) {
    h <- yunque::layer_norm(x, ly$attn_ln$w, ly$attn_ln$b, eps)
    q <- .yq_heads(yunque::linear(h, ly$q$w, ly$q$b), batch, nh, hd)
    k <- .yq_heads(yunque::linear(h, ly$k$w, ly$k$b), batch, nh, hd)
    v <- .yq_heads(yunque::linear(h, ly$v$w, ly$v$b), batch, nh, hd)
    a <- yunque::sdpa(q, k, v)
    a <- anvl::nv_reshape(anvl::nv_transpose(a, c(1L, 3L, 2L, 4L)),
      c(batch, seq, ns))
    x <- x + yunque::linear(a, ly$out$w, ly$out$b)

    h2 <- yunque::layer_norm(x, ly$mlp_ln$w, ly$mlp_ln$b, eps)
    m <- yunque::linear(
      yunque::gelu(yunque::linear(h2, ly$fc1$w, ly$fc1$b), "none"),
      ly$fc2$w, ly$fc2$b)
    x <- x + m
  }
  yunque::layer_norm(x, w$ln_post$w, w$ln_post$b, eps)
}
