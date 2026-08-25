# Prefix-conditioned decode termination: with committed tokens extending
# the prompt, first-step blank/EOT suppression must not apply, or a
# decode of an already-complete utterance is forced to hallucinate one
# more token per tick. Uses a fake model whose highest-probability next
# token is always EOT -- no weights, no download.

if (!requireNamespace("torch", quietly = TRUE) ||
    !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

special <- whisper:::whisper_special_tokens("tiny")
nv <- 51865L
device <- torch::torch_device("cpu")

fake_model <- list(
  decode = function(tokens, encoder_output, kv_cache = NULL,
                    need_weights = FALSE) {
    n <- tokens$size(2)
    logits <- torch::torch_zeros(1L, n, nv)
    logits[1, n, special$eot + 1L] <- 10
    list(logits = logits, kv_cache = NULL)
  }
)
fake_tokenizer <- list(
  model = "tiny",
  suppress_tokens = integer(0),
  blank_tokens = c(220L, special$eot), # space + EOT, as built for real
  decode = function(ids) ""
)

initial <- whisper:::get_initial_tokens("en", "transcribe",
  model = "tiny", timestamps = FALSE)
start <- c(initial, 3000L) # SOT sequence + one committed content token
tok_tensor <- torch::torch_tensor(matrix(start, nrow = 1L),
  dtype = torch::torch_long(), device = device)

# suppress_blank = FALSE: the continuation terminates immediately at EOT
r_free <- whisper:::greedy_decode(fake_model, NULL, tok_tensor,
  fake_tokenizer, max_length = length(start) + 5L,
  suppress_blank = FALSE, device = device)
expect_equal(length(r_free$tokens), length(start))

# suppress_blank = TRUE (the batch default): EOT is masked on the first
# generated step, so the decode is forced past the natural stop -- which
# is exactly why streaming must turn it off once anything is committed
r_blocked <- whisper:::greedy_decode(fake_model, NULL, tok_tensor,
  fake_tokenizer, max_length = length(start) + 5L,
  suppress_blank = TRUE, device = device)
expect_true(length(r_blocked$tokens) > length(start))

# And the flag threads through decode_with_fallback
r_dwf <- whisper:::decode_with_fallback(fake_model, NULL, tok_tensor,
  fake_tokenizer, temperatures = 0, beam_size = 1L, best_of = 1L,
  max_length = length(start) + 5L, suppress_blank = FALSE,
  jit = FALSE, device = device)
expect_equal(length(r_dwf$tokens), length(start))
