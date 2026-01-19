# Tests for decoder

# Skip if torch not fully installed (libtorch binaries required)
if (!requireNamespace("torch", quietly = TRUE) ||
    !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

# Test decoder layer creation
expect_silent(layer <- whisper:::whisper_decoder_layer(384L, 6L))

# Test full decoder creation
cfg <- whisper_config("tiny")
expect_silent(decoder <- whisper:::create_decoder(cfg))

# Test forward pass with dummy input
batch_size <- 1L
seq_len <- 5L
src_len <- 1500L

tokens <- torch::torch_randint(0, cfg$n_vocab - 1, c(batch_size, seq_len),
  dtype = torch::torch_long())
encoder_output <- torch::torch_randn(batch_size, src_len, cfg$n_text_state)

# Forward pass
decoder$eval()
torch::with_no_grad({
  result <- decoder(tokens, encoder_output)
})

# Check output shape
expect_equal(result$hidden_states$size(1), batch_size)
expect_equal(result$hidden_states$size(2), seq_len)
expect_equal(result$hidden_states$size(3), cfg$n_text_state)

# Test logits
logits <- decoder$get_logits(result$hidden_states)
expect_equal(logits$size(1), batch_size)
expect_equal(logits$size(2), seq_len)
expect_equal(logits$size(3), cfg$n_vocab)
