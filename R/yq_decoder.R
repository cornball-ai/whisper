# anvl/yunque port of the Whisper text decoder (self + cross attention).
# Full-forward (no KV cache) path first: parity-checkable against torch;
# the cache + greedy loop layer on top later.

# Multi-head attention block: q from `xq`, k/v from `xkv` (self-attn passes
# the same tensor for both; cross-attn passes decoder x and encoder output).
.yq_mha <- function(xq, xkv, wq, wk, wv, wo, n_head, head_dim, mask = NULL) {
  s <- anvl::shape(xq)
  batch <- s[1L]
  seq <- s[2L]
  ns <- s[3L]
  q <- .yq_heads(yunque::linear(xq, wq$w, wq$b), batch, n_head, head_dim)
  k <- .yq_heads(yunque::linear(xkv, wk$w, wk$b), batch, n_head, head_dim)
  v <- .yq_heads(yunque::linear(xkv, wv$w, wv$b), batch, n_head, head_dim)
  a <- yunque::sdpa(q, k, v, mask = mask)
  a <- anvl::nv_reshape(anvl::nv_transpose(a, c(1L, 3L, 2L, 4L)),
    c(batch, seq, ns))
  yunque::linear(a, wo$w, wo$b)
}

#' Load Whisper decoder weights (anvl)
#'
#' @param path Path to model.safetensors.
#' @param config A \code{\link{whisper_config}} list.
#' @param prefix Key prefix (HF checkpoints use \code{"model.decoder."}).
#'
#' @return A named list of weights (token/pos embeddings as R matrices for
#'   host-side lookup; the tied logit projection and layer weights as anvl
#'   arrays) plus \code{config}.
#'
#' @export
yq_decoder_load_weights <- function(path, config, prefix = "model.decoder.") {
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

  layers <- lapply(seq_len(config$n_text_layer) - 1L, function(i) {
    p <- paste0("layers.", i, ".")
    list(
      attn_ln = ln(paste0(p, "self_attn_layer_norm")),
      q = lin(paste0(p, "self_attn.q_proj")),
      k = lin(paste0(p, "self_attn.k_proj")),
      v = lin(paste0(p, "self_attn.v_proj")),
      out = lin(paste0(p, "self_attn.out_proj")),
      cross_ln = ln(paste0(p, "encoder_attn_layer_norm")),
      cq = lin(paste0(p, "encoder_attn.q_proj")),
      ck = lin(paste0(p, "encoder_attn.k_proj")),
      cv = lin(paste0(p, "encoder_attn.v_proj")),
      cout = lin(paste0(p, "encoder_attn.out_proj")),
      mlp_ln = ln(paste0(p, "final_layer_norm")),
      fc1 = lin(paste0(p, "fc1")),
      fc2 = lin(paste0(p, "fc2"))
    )
  })
  # token embedding stays an R matrix [n_vocab, n_state] for host-side gather;
  # its transpose is the tied logit projection [n_state, n_vocab].
  token_emb <- yunque::st_read(st, paste0(prefix, "embed_tokens.weight"))
  pos_emb <- yunque::st_read(st, paste0(prefix, "embed_positions.weight"))
  list(
    token_emb = token_emb,
    pos_emb = pos_emb,
    logit_w = anvl::nv_array(t(token_emb), dtype = "f32"),
    layers = layers,
    ln = ln("layer_norm"),
    config = config
  )
}

#' Whisper decoder full forward (anvl, no cache)
#'
#' Torch-free port of \code{whisper_decoder$forward} + \code{get_logits}
#' for a fixed prompt: token + learned-position embeddings, pre-norm
#' layers (causal self-attn + cross-attn to the encoder output + GELU
#' MLP), final LayerNorm, and weight-tied logits. Autoregressive KV
#' caching is layered on separately.
#'
#' @param tokens Integer matrix \code{[batch, seq]} of 0-based token ids.
#' @param xa AnvlArray \code{[batch, src, n_state]} encoder output.
#' @param w Weights from \code{\link{yq_decoder_load_weights}}.
#'
#' @return AnvlArray logits \code{[batch, seq, n_vocab]}.
#'
#' @export
yq_decoder <- function(tokens, xa, w) {
  cfg <- w$config
  nh <- cfg$n_text_head
  hd <- cfg$n_text_state %/% nh
  eps <- 1e-5
  tokens <- matrix(as.integer(tokens), nrow = nrow(tokens))
  seq <- ncol(tokens)

  x <- yunque::embedding(w$token_emb, tokens) # [B, S, n_state]
  pos <- yunque::embedding(w$pos_emb, seq_len(seq) - 1L) # [S, n_state]
  x <- x + anvl::nv_broadcast_to(anvl::nv_unsqueeze(pos, 1L), anvl::shape(x))

  # causal mask [1, 1, S, S]: -1e9 above the diagonal, 0 on/below
  mr <- matrix(0, seq, seq)
  mr[upper.tri(mr)] <- -1e9
  mask <- anvl::nv_array(array(mr, c(1L, 1L, seq, seq)), dtype = "f32")

  for (ly in w$layers) {
    h <- yunque::layer_norm(x, ly$attn_ln$w, ly$attn_ln$b, eps)
    x <- x + .yq_mha(h, h, ly$q, ly$k, ly$v, ly$out, nh, hd, mask = mask)

    h <- yunque::layer_norm(x, ly$cross_ln$w, ly$cross_ln$b, eps)
    x <- x + .yq_mha(h, xa, ly$cq, ly$ck, ly$cv, ly$cout, nh, hd)

    h <- yunque::layer_norm(x, ly$mlp_ln$w, ly$mlp_ln$b, eps)
    x <- x + yunque::linear(
      yunque::gelu(yunque::linear(h, ly$fc1$w, ly$fc1$b), "none"),
      ly$fc2$w, ly$fc2$b)
  }
  x <- yunque::layer_norm(x, w$ln$w, w$ln$b, eps)
  yunque::linear(x, w$logit_w)
}

# Cross-attention K/V per layer, computed once from the encoder output
# (constant across all decode steps).
.yq_cross_kv <- function(xa, w) {
  cfg <- w$config
  nh <- cfg$n_text_head
  hd <- cfg$n_text_state %/% nh
  batch <- anvl::shape(xa)[1L]
  lapply(w$layers, function(ly) list(
    k = .yq_heads(yunque::linear(xa, ly$ck$w, ly$ck$b), batch, nh, hd),
    v = .yq_heads(yunque::linear(xa, ly$cv$w, ly$cv$b), batch, nh, hd)
  ))
}

#' Incremental (KV-cached) decoder forward (anvl)
#'
#' The autoregressive form of \code{\link{yq_decoder}}: \code{self_kv} NULL
#' is the prompt prefill (causal mask); non-NULL processes new token(s)
#' attending over the cached keys (all at earlier positions, so no mask).
#' Cross-attention reuses \code{cross_kv} from \code{.yq_cross_kv}. Turns
#' the O(S^2) recompute into O(S).
#'
#' @param tokens Integer matrix \code{[batch, seq]} of 0-based token ids.
#' @param cross_kv Per-layer cross K/V from \code{.yq_cross_kv}.
#' @param self_kv Per-layer self K/V cache, or NULL to prefill.
#' @param w Weights from \code{\link{yq_decoder_load_weights}}.
#' @param offset Integer. Absolute position of the first token.
#'
#' @return List with \code{logits} \code{[batch, seq, n_vocab]} and the
#'   extended \code{self_kv}.
#'
#' @export
yq_decoder_incremental <- function(tokens, cross_kv, self_kv, w, offset) {
  cfg <- w$config
  nh <- cfg$n_text_head
  hd <- cfg$n_text_state %/% nh
  ns <- cfg$n_text_state
  eps <- 1e-5
  tokens <- matrix(as.integer(tokens), nrow = nrow(tokens))
  batch <- nrow(tokens)
  seq <- ncol(tokens)

  x <- yunque::embedding(w$token_emb, tokens)
  pos <- yunque::embedding(w$pos_emb, offset + seq_len(seq) - 1L)
  x <- x + anvl::nv_broadcast_to(anvl::nv_unsqueeze(pos, 1L), anvl::shape(x))

  mask <- NULL
  if (is.null(self_kv)) {
    mr <- matrix(0, seq, seq)
    mr[upper.tri(mr)] <- -1e9
    mask <- anvl::nv_array(array(mr, c(1L, 1L, seq, seq)), dtype = "f32")
  }
  finish <- function(a) anvl::nv_reshape(
    anvl::nv_transpose(a, c(1L, 3L, 2L, 4L)), c(batch, seq, ns))

  new_self_kv <- vector("list", length(w$layers))
  for (li in seq_along(w$layers)) {
    ly <- w$layers[[li]]
    h <- yunque::layer_norm(x, ly$attn_ln$w, ly$attn_ln$b, eps)
    q <- .yq_heads(yunque::linear(h, ly$q$w, ly$q$b), batch, nh, hd)
    kk <- .yq_heads(yunque::linear(h, ly$k$w, ly$k$b), batch, nh, hd)
    vv <- .yq_heads(yunque::linear(h, ly$v$w, ly$v$b), batch, nh, hd)
    if (!is.null(self_kv)) {
      kk <- anvl::nv_concatenate(self_kv[[li]]$k, kk, dimension = 3L)
      vv <- anvl::nv_concatenate(self_kv[[li]]$v, vv, dimension = 3L)
    }
    new_self_kv[[li]] <- list(k = kk, v = vv)
    x <- x + yunque::linear(finish(yunque::sdpa(q, kk, vv, mask = mask)),
      ly$out$w, ly$out$b)

    h <- yunque::layer_norm(x, ly$cross_ln$w, ly$cross_ln$b, eps)
    q <- .yq_heads(yunque::linear(h, ly$cq$w, ly$cq$b), batch, nh, hd)
    x <- x + yunque::linear(
      finish(yunque::sdpa(q, cross_kv[[li]]$k, cross_kv[[li]]$v)),
      ly$cout$w, ly$cout$b)

    h <- yunque::layer_norm(x, ly$mlp_ln$w, ly$mlp_ln$b, eps)
    x <- x + yunque::linear(
      yunque::gelu(yunque::linear(h, ly$fc1$w, ly$fc1$b), "none"),
      ly$fc2$w, ly$fc2$b)
  }
  x <- yunque::layer_norm(x, w$ln$w, w$ln$b, eps)
  list(logits = yunque::linear(x, w$logit_w), self_kv = new_self_kv)
}
