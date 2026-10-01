# Tests for the cuDNN-free encoder conv stem

if (!requireNamespace("torch", quietly = TRUE) ||
    !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

torch::torch_manual_seed(1)
unfold_conv <- whisper:::conv1d_unfold

# Matches nnf_conv1d for both stem convs, even and odd lengths
for (len in c(3000L, 101L)) {
  x <- torch::torch_randn(1L, 80L, len)
  w <- torch::torch_randn(64L, 80L, 3L)
  b <- torch::torch_randn(64L)
  ref <- torch::nnf_conv1d(x, w, b, padding = 1L)
  out <- unfold_conv(x, w, b)
  expect_equal(out$shape, ref$shape)
  expect_true((out - ref)$abs()$max()$item() < 1e-4)

  x2 <- torch::torch_randn(1L, 64L, len)
  w2 <- torch::torch_randn(64L, 64L, 3L)
  ref2 <- torch::nnf_conv1d(x2, w2, b, stride = 2L, padding = 1L)
  out2 <- unfold_conv(x2, w2, b, stride = 2L)
  expect_equal(out2$shape, ref2$shape)
  expect_true((out2 - ref2)$abs()$max()$item() < 1e-4)
}

# Batch > 1
x <- torch::torch_randn(2L, 8L, 20L)
w <- torch::torch_randn(4L, 8L, 3L)
b <- torch::torch_randn(4L)
expect_true((unfold_conv(x, w, b, stride = 2L) -
  torch::nnf_conv1d(x, w, b, stride = 2L, padding = 1L))$abs()$max()$item() <
  1e-4)

# Only fp16 on CUDA can need it
expect_false(whisper:::.stem_needs_unfold(torch::torch_randn(1L, 4L, 4L)))
expect_false(whisper:::.stem_needs_unfold(
  torch::torch_randn(1L, 4L, 4L, dtype = torch::torch_float16())))

# The encoder's unfold path gives the cuDNN path's output
enc <- whisper:::whisper_encoder(80L, 1500L, 64L, 2L, 1L)
enc$eval()
mel <- torch::torch_randn(1L, 80L, 3000L)
old <- options(whisper.jit = FALSE)
torch::with_no_grad(ref <- enc(mel))
out <- NULL
local({
  ns <- asNamespace("whisper")
  orig <- get(".stem_needs_unfold", ns)
  unlockBinding(".stem_needs_unfold", ns)
  on.exit({
    assign(".stem_needs_unfold", orig, ns)
    lockBinding(".stem_needs_unfold", ns)
  })
  assign(".stem_needs_unfold", function(x) TRUE, ns)
  torch::with_no_grad(out <<- enc(mel))
})
options(old)
expect_true((out - ref)$abs()$max()$item() < 1e-4)
