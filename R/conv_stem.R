# Encoder conv stem without cuDNN.
#
# On the GTX 16-series (TU116/TU117) cuDNN's fp16 convolution returns NaN
# once a conv is large enough, and the encoder's stride-2 conv2 always is
# (measured on a GTX 1660 Ti, cuDNN 9.8). cuDNN's benchmark and
# deterministic modes fail the same way. Everything else whisper runs in
# fp16 (matmul, attention, norms) is correct on these cards, so only the
# stem moves off cuDNN. R torch has no public switch to skip cuDNN for one
# call, so the conv is computed as a matmul over unfolded windows.

# conv1d as unfold + matmul.
# x (N, C_in, L), weight (C_out, C_in, K), bias (C_out) -> (N, C_out, L_out)
conv1d_unfold <- function(
  x,
  weight,
  bias,
  stride = 1L,
  padding = 1L
) {
  k <- weight$size(3)
  x <- torch::nnf_pad(x, c(padding, padding))
  # (N, C_in, L_out, K) -> (N, L_out, C_in * K), C_in-major like the weight
  cols <- x$unfold(-1L, k, stride)$permute(c(1L, 3L, 2L, 4L))
  cols <- cols$flatten(start_dim = 3L)
  out <- torch::nnf_linear(cols, weight$flatten(start_dim = 2L), bias)
  out$transpose(2L, 3L)
}

# Whether the stem must avoid cuDNN for this input: fp16 on a GTX 16-series.
.stem_needs_unfold <- function(x) {
  x$device$type == "cuda" && x$dtype == torch::torch_float16() &&
    .fp16_broken_gpu(x$device)
}
