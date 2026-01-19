# Tests for encoder

# Skip if torch not fully installed (libtorch binaries required)
if (!requireNamespace("torch", quietly = TRUE) ||
    !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

# Test attention module creation
expect_silent(attn <- whisper:::whisper_attention(384L, 6L))

# Test encoder layer creation
expect_silent(layer <- whisper:::whisper_encoder_layer(384L, 6L))

# Test full encoder creation
cfg <- whisper_config("tiny")
expect_silent(encoder <- whisper:::create_encoder(cfg))

# Test forward pass with dummy input
batch_size <- 1L
n_mels <- 80L
n_frames <- 3000L # 30 seconds at 100 fps

mel <- torch::torch_randn(batch_size, n_mels, n_frames)

# Forward pass (eval mode to avoid batchnorm issues)
encoder$eval()
torch::with_no_grad({
  output <- encoder(mel)
})

# Check output shape
# After conv2 with stride=2: n_frames/2 = 1500
expect_equal(output$size(1), batch_size)
expect_equal(output$size(2), 1500L) # n_audio_ctx
expect_equal(output$size(3), cfg$n_audio_state)
