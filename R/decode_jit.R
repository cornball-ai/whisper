# Greedy decoding via a jit_compile'd TorchScript decode step.
#
# Each generated token's full decoder forward (n_layer x {self-attn,
# cross-attn, FFN} + final LayerNorm) runs as ONE TorchScript function
# inside libtorch. R keeps the eager prefill, the sampling/timestamp
# logic, and the loop shell. The motivation is identical to chatterbox's
# t3_inference_jit (see chatterbox/R/t3_jit.R): even optimally-written
# eager R keeps a per-op R->lantern dispatch floor (~190us x hundreds of
# ops/token); collapsing the per-token forward into a single TorchScript
# call removes that floor without compiled code or linking against
# torch's private libraries.
#
# Whisper's step differs from a Llama step in three ways: LayerNorm (with
# bias) rather than RMSNorm, biases on every projection, and a
# cross-attention block whose K/V are computed once from the encoder
# output and cached. The self-attention K/V cache is pre-allocated and
# filled at the current position each step; the cross K/V are passed in
# as stacked (n_layer, ...) tensors.
#
# Correctness is verified token-for-token against the eager greedy_decode
# path (inst/tinytest/test_decode_jit.R).

# Session cache for compiled decode steps (keyed by architecture)
.whisper_jit_decode_cache <- new.env(parent = emptyenv())

#' Build and compile the TorchScript decode step for an architecture
#'
#' @param n_layers,n_heads,head_dim,eps Decoder architecture parameters
#' @return The compiled script function
#' @noRd
.get_whisper_jit_decode_step <- function(n_layers, n_heads, head_dim, eps) {
  key <- paste(n_layers, n_heads, head_dim, eps, sep = "_")
  if (!is.null(.whisper_jit_decode_cache[[key]])) {
    return(.whisper_jit_decode_cache[[key]])
  }

  # ATen-builtin calls only (torch.matmul/torch.gelu/torch.layer_norm are
  # resolvable in this lantern's TorchScript environment; the
  # torch.nn.functional namespace is not). LayerNorm is done by hand in
  # float for fp16 parity, matching torch's eager nn_layer_norm.
  src <- sprintf("
def ln(x: Tensor, w: Tensor, b: Tensor, eps: float) -> Tensor:
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = (xf - mean).pow(2).mean(-1, keepdim=True)
    normed = (xf - mean) * torch.rsqrt(var + eps)
    return (normed * w + b).to(x.dtype)

def decode_step(x: Tensor, w: List[Tensor], gw: List[Tensor],
                k_cache: Tensor, v_cache: Tensor, ck: Tensor, cv: Tensor,
                pos: int, valid: int) -> Tensor:
    n_layers = %d
    n_heads = %d
    head_dim = %d
    eps = %s
    B = x.size(0)
    for i in range(n_layers):
        b = i * 21
        # --- self-attention (causal, KV-cached) ---
        resid = x
        normed = ln(x, w[b], w[b + 1], eps)
        q = (torch.matmul(normed, w[b + 2].t()) + w[b + 3]).view(B, 1, n_heads, head_dim).transpose(1, 2)
        k = torch.matmul(normed, w[b + 4].t()).view(B, 1, n_heads, head_dim).transpose(1, 2)
        v = (torch.matmul(normed, w[b + 5].t()) + w[b + 6]).view(B, 1, n_heads, head_dim).transpose(1, 2)
        k_cache[i, :, :, pos] = k.squeeze(2)
        v_cache[i, :, :, pos] = v.squeeze(2)
        kk = k_cache[i, :, :, :valid]
        vv = v_cache[i, :, :, :valid]
        attn = torch.scaled_dot_product_attention(q, kk, vv)
        attn = attn.transpose(1, 2).reshape(B, 1, n_heads * head_dim)
        x = resid + torch.matmul(attn, w[b + 7].t()) + w[b + 8]
        # --- cross-attention (encoder K/V cached) ---
        resid = x
        normed = ln(x, w[b + 9], w[b + 10], eps)
        cq = (torch.matmul(normed, w[b + 11].t()) + w[b + 12]).view(B, 1, n_heads, head_dim).transpose(1, 2)
        cattn = torch.scaled_dot_product_attention(cq, ck[i], cv[i])
        cattn = cattn.transpose(1, 2).reshape(B, 1, n_heads * head_dim)
        x = resid + torch.matmul(cattn, w[b + 13].t()) + w[b + 14]
        # --- feed-forward ---
        resid = x
        normed = ln(x, w[b + 15], w[b + 16], eps)
        h = torch.gelu(torch.matmul(normed, w[b + 17].t()) + w[b + 18])
        h = torch.matmul(h, w[b + 19].t()) + w[b + 20]
        x = resid + h
    return ln(x, gw[0], gw[1], eps)
",
    n_layers, n_heads, head_dim, format(eps, scientific = FALSE))

  cu <- torch::jit_compile(src)
  .whisper_jit_decode_cache[[key]] <- cu$decode_step
  cu$decode_step
}

#' Build the decode step that also returns cross-attention weights
#'
#' Same as the plain step, but cross-attention is computed manually (softmax)
#' instead of via fused SDPA, so the per-layer attention weights are exposed
#' for word-timestamp DTW alignment. Returns (hidden, stacked_xattn) where
#' stacked_xattn is (n_layers, B, n_head, 1, src_len). Self-attention keeps
#' SDPA (its weights are not needed). scale = sqrt(head_dim), matching the
#' eager need_weights path.
#'
#' @param n_layers,n_heads,head_dim,eps Decoder architecture parameters
#' @return The compiled script function returning a (hidden, xattn) tuple
#' @noRd
.get_whisper_jit_decode_step_xattn <- function(n_layers, n_heads, head_dim, eps) {
  key <- paste("x", n_layers, n_heads, head_dim, eps, sep = "_")
  if (!is.null(.whisper_jit_decode_cache[[key]])) {
    return(.whisper_jit_decode_cache[[key]])
  }
  src <- sprintf("
def ln(x: Tensor, w: Tensor, b: Tensor, eps: float) -> Tensor:
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = (xf - mean).pow(2).mean(-1, keepdim=True)
    normed = (xf - mean) * torch.rsqrt(var + eps)
    return (normed * w + b).to(x.dtype)

def decode_step_x(x: Tensor, w: List[Tensor], gw: List[Tensor],
                  k_cache: Tensor, v_cache: Tensor, ck: Tensor, cv: Tensor,
                  pos: int, valid: int) -> Tuple[Tensor, Tensor]:
    n_layers = %d
    n_heads = %d
    head_dim = %d
    eps = %s
    scale = %s
    B = x.size(0)
    xws : List[Tensor] = []
    for i in range(n_layers):
        b = i * 21
        # --- self-attention (causal, KV-cached, SDPA) ---
        resid = x
        normed = ln(x, w[b], w[b + 1], eps)
        q = (torch.matmul(normed, w[b + 2].t()) + w[b + 3]).view(B, 1, n_heads, head_dim).transpose(1, 2)
        k = torch.matmul(normed, w[b + 4].t()).view(B, 1, n_heads, head_dim).transpose(1, 2)
        v = (torch.matmul(normed, w[b + 5].t()) + w[b + 6]).view(B, 1, n_heads, head_dim).transpose(1, 2)
        k_cache[i, :, :, pos] = k.squeeze(2)
        v_cache[i, :, :, pos] = v.squeeze(2)
        attn = torch.scaled_dot_product_attention(q, k_cache[i, :, :, :valid], v_cache[i, :, :, :valid])
        attn = attn.transpose(1, 2).reshape(B, 1, n_heads * head_dim)
        x = resid + torch.matmul(attn, w[b + 7].t()) + w[b + 8]
        # --- cross-attention (manual softmax, weights exposed) ---
        resid = x
        normed = ln(x, w[b + 9], w[b + 10], eps)
        cq = (torch.matmul(normed, w[b + 11].t()) + w[b + 12]).view(B, 1, n_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(cq, ck[i].transpose(2, 3)) / scale
        cw = torch.softmax(scores, -1)
        xws.append(cw)
        cattn = torch.matmul(cw, cv[i]).transpose(1, 2).reshape(B, 1, n_heads * head_dim)
        x = resid + torch.matmul(cattn, w[b + 13].t()) + w[b + 14]
        # --- feed-forward ---
        resid = x
        normed = ln(x, w[b + 15], w[b + 16], eps)
        h = torch.gelu(torch.matmul(normed, w[b + 17].t()) + w[b + 18])
        h = torch.matmul(h, w[b + 19].t()) + w[b + 20]
        x = resid + h
    return (ln(x, gw[0], gw[1], eps), torch.stack(xws))
",
    n_layers, n_heads, head_dim, format(eps, scientific = FALSE),
    format(sqrt(head_dim), nsmall = 1))

  cu <- torch::jit_compile(src)
  .whisper_jit_decode_cache[[key]] <- cu$decode_step_x
  cu$decode_step_x
}

#' Extract per-layer decoder weights in decode-step order
#'
#' 21 tensors per layer, in the order the TorchScript step indexes them:
#' self-attn LayerNorm (w,b), Q (w,b), K (w; no bias), V (w,b), out (w,b);
#' cross-attn LayerNorm (w,b), Q (w,b), out (w,b); FFN LayerNorm (w,b),
#' fc1 (w,b), fc2 (w,b). Tensors are borrowed by reference - no copies.
#'
#' @param decoder WhisperDecoder module
#' @return Flat list of n_layer * 21 tensors
#' @noRd
.get_whisper_layer_weights <- function(decoder) {
  n_layers <- length(decoder$blocks)
  layers <- vector("list", n_layers)
  for (i in seq_len(n_layers)) {
    layer <- decoder$blocks[[i]]
    layers[[i]] <- list(
      layer$attn_ln$weight, layer$attn_ln$bias,
      layer$attn$query$weight, layer$attn$query$bias,
      layer$attn$key$weight,
      layer$attn$value$weight, layer$attn$value$bias,
      layer$attn$out$weight, layer$attn$out$bias,
      layer$cross_attn_ln$weight, layer$cross_attn_ln$bias,
      layer$cross_attn$query$weight, layer$cross_attn$query$bias,
      layer$cross_attn$out$weight, layer$cross_attn$out$bias,
      layer$mlp_ln$weight, layer$mlp_ln$bias,
      layer$mlp[[1]]$weight, layer$mlp[[1]]$bias,
      layer$mlp[[3]]$weight, layer$mlp[[3]]$bias
    )
  }
  do.call(c, layers)
}

#' Greedy Decoding with a TorchScript decode loop
#'
#' Token-for-token equivalent of \code{\link{greedy_decode}}: eager
#' prefill on the initial prompt, then each new token's decoder forward
#' runs as one \code{jit_compile}'d TorchScript call. The self-attention
#' KV cache is pre-allocated to \code{max_length} and the cross-attention
#' K/V are cached once from the encoder output. When \code{word_timestamps}
#' is TRUE it uses the cross-attention-weight variant of the step (manual
#' softmax cross-attention) and collects the per-token weights in the same
#' order as the eager path, so word-level DTW alignment works on the JIT
#' path too.
#'
#' @param model WhisperModel
#' @param encoder_output Encoder hidden states
#' @param initial_tokens Initial token tensor (batch=1)
#' @param tokenizer Tokenizer
#' @param max_length Maximum output length
#' @param timestamps Whether to allow timestamp tokens
#' @param word_timestamps Whether to collect cross-attention weights
#' @param device Device
#' @return List with tokens, cross_attn_weights, sum_logprob, n_tokens
#' @keywords internal
greedy_decode_jit <- function(
  model,
  encoder_output,
  initial_tokens,
  tokenizer,
  max_length = 224L,
  timestamps = FALSE,
  word_timestamps = FALSE,
  device
) {
  special <- whisper_special_tokens(tokenizer$model)
  decoder <- model$decoder
  n_layers <- length(decoder$blocks)
  n_heads <- decoder$blocks[[1]]$attn$n_head
  head_dim <- decoder$blocks[[1]]$attn$head_dim
  eps <- 1e-5

  need_w <- isTRUE(word_timestamps)
  step_fn <- if (need_w) {
    .get_whisper_jit_decode_step_xattn(n_layers, n_heads, head_dim, eps)
  } else {
    .get_whisper_jit_decode_step(n_layers, n_heads, head_dim, eps)
  }
  wflat <- .get_whisper_layer_weights(decoder)
  gw <- list(decoder$ln$weight, decoder$ln$bias)
  tok_emb_w <- decoder$token_embedding$weight
  pos_emb_w <- decoder$positional_embedding$weight

  generated <- as.integer(as.array(initial_tokens$cpu()))
  sample_begin <- length(generated)
  cond_len <- sample_begin
  sum_logprob <- 0
  n_tokens <- 0L
  all_cross_attn <- if (need_w) list() else NULL

  torch::with_no_grad({
    # Eager prefill on the initial prompt: gives the first logits and the
    # self + cross KV caches (and, for word timestamps, the prompt's
    # cross-attention weights -- the eager path's first collected step).
    result <- model$decode(initial_tokens, encoder_output, kv_cache = NULL,
      need_weights = need_w)
    kv <- result$kv_cache
    dtype <- tok_emb_w$dtype
    # cur_xattn holds the cross-attn weights of the forward that produced
    # the current next_logits (prompt prefill to start).
    cur_xattn <- if (need_w) result$cross_attn_weights else NULL

    # Stack cross-attention K/V (constant for the whole generation)
    ck <- torch::torch_stack(lapply(kv, function(l) l$cross$k), dim = 1L)
    cv <- torch::torch_stack(lapply(kv, function(l) l$cross$v), dim = 1L)

    # Pre-allocate the self-attention KV cache and seed it from prefill
    k_cache <- torch::torch_zeros(n_layers, 1L, n_heads, max_length, head_dim,
      device = device, dtype = dtype)
    v_cache <- torch::torch_zeros_like(k_cache)
    for (l in seq_len(n_layers)) {
      k_cache[l, , , 1:cond_len, ] <- kv[[l]]$self$k
      v_cache[l, , , 1:cond_len, ] <- kv[[l]]$self$v
    }

    logits <- result$logits
    no_speech_prob <- .no_speech_prob(logits, generated, special)
    next_logits <- logits[, logits$size(2), ]
    # Suppression masks (SuppressTokens / SuppressBlank), same as the eager
    # path so JIT and eager stay token-for-token equivalent.
    nv <- next_logits$size(2)
    supp_mask <- .suppress_mask(tokenizer$suppress_tokens, nv, device,
      next_logits$dtype)
    blank_mask <- .suppress_mask(tokenizer$blank_tokens, nv, device,
      next_logits$dtype)
    pos <- cond_len  # 0-based position of the next token to generate

    for (i in seq_len(max_length)) {
      if (length(generated) >= max_length) break

      next_logits <- next_logits + supp_mask
      if (length(generated) == sample_begin) {
        next_logits <- next_logits + blank_mask
      }
      if (timestamps) {
        next_logits <- apply_timestamp_rules(next_logits, generated,
          special, sample_begin)
      }

      log_probs <- torch::nnf_log_softmax(next_logits, dim = -1L)
      next_token <- next_logits$argmax(dim = -1L)
      next_token_id <- as.integer(next_token$item()) - 1L

      sum_logprob <- sum_logprob +
        as.numeric(log_probs[1, next_token$item()]$item())
      n_tokens <- n_tokens + 1L

      if (next_token_id == special$eot) break

      generated <- c(generated, next_token_id)
      # cur_xattn is the cross-attn of the forward that predicted this token
      # (matching the eager path's per-step collection order).
      if (need_w) {
        all_cross_attn <- c(all_cross_attn, list(cur_xattn))
      }
      if (length(generated) >= max_length) break

      # Embed the new token at its absolute position (both lookups
      # 1-indexed: token id + 1, position + 1) and run the decoder step.
      x <- (tok_emb_w[next_token_id + 1L, ] +
        pos_emb_w[pos + 1L, ])$view(c(1L, 1L, -1L))
      if (need_w) {
        out <- step_fn(x, wflat, gw, k_cache, v_cache, ck, cv,
          torch::jit_scalar(pos), torch::jit_scalar(pos + 1L))
        hidden <- out[[1]]
        xw <- out[[2]]  # (n_layers, B, n_head, 1, src); split to per-layer list
        cur_xattn <- lapply(seq_len(n_layers), function(l) xw[l, , , , ])
      } else {
        hidden <- step_fn(x, wflat, gw, k_cache, v_cache, ck, cv,
          torch::jit_scalar(pos), torch::jit_scalar(pos + 1L))
      }
      logits <- torch::torch_matmul(hidden, tok_emb_w$t())
      next_logits <- logits[, logits$size(2), ]
      pos <- pos + 1L
    }
  })

  list(
    tokens = generated,
    cross_attn_weights = all_cross_attn,
    sum_logprob = sum_logprob,
    n_tokens = n_tokens,
    no_speech_prob = no_speech_prob
  )
}
