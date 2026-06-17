# The TorchScript greedy decode step must produce the same tokens as the
# eager greedy_decode path. CUDA-only (the JIT path is gated to CUDA) and
# needs the tiny weights, so this runs locally (at_home) only.

if (at_home() &&
    requireNamespace("torch", quietly = TRUE) &&
    torch::torch_is_installed() &&
    torch::cuda_is_available() &&
    whisper:::model_exists("tiny")) {

  dev <- torch::torch_device("cuda")
  dtype <- torch::torch_float16()
  model <- whisper:::load_whisper_model("tiny", device = dev, dtype = dtype,
    verbose = FALSE)
  tok <- whisper:::whisper_tokenizer("tiny")
  cfg <- whisper:::whisper_config("tiny")
  audio <- system.file("audio", "jfk.mp3", package = "whisper")

  mel <- whisper:::audio_to_mel(audio, n_mels = cfg$n_mels, device = dev,
    dtype = dtype)
  init <- whisper:::get_initial_tokens("en", "transcribe", model = "tiny",
    timestamps = FALSE)
  init_t <- torch::torch_tensor(matrix(init, nrow = 1),
    dtype = torch::torch_long(), device = dev)
  torch::with_no_grad({
    enc <- model$encode(mel)
  })

  # Without timestamp constraints
  eager <- whisper:::greedy_decode(model, enc, init_t, tok,
    max_length = cfg$n_text_ctx, timestamps = FALSE, device = dev)
  jit <- whisper:::greedy_decode_jit(model, enc, init_t, tok,
    max_length = cfg$n_text_ctx, timestamps = FALSE, device = dev)
  expect_identical(jit$tokens, eager$tokens,
    info = "jit greedy decode matches eager (no timestamps)")

  # With timestamp logit rules
  eager_ts <- whisper:::greedy_decode(model, enc, init_t, tok,
    max_length = cfg$n_text_ctx, timestamps = TRUE, device = dev)
  jit_ts <- whisper:::greedy_decode_jit(model, enc, init_t, tok,
    max_length = cfg$n_text_ctx, timestamps = TRUE, device = dev)
  expect_identical(jit_ts$tokens, eager_ts$tokens,
    info = "jit greedy decode matches eager (timestamps)")
}
