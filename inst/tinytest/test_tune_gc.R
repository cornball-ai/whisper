# Tests for whisper_tune_gc() and its CUDA-free device/dtype resolution.
#
# The behaviour under test is a negative one: this function must NOT
# initialize CUDA. torch reads the allocator rates once, at CUDA init, so a
# tuner that probes first sets them after they were read -- a silent no-op
# that still prints a success message. Everything here therefore runs
# without torch being usable, which is also why these tests need no GPU.

if (!requireNamespace("torch", quietly = TRUE)) {
  exit_file("torch not installed")
}

# ---- device resolution, no torch involved ----

idx <- whisper:::.gc_device_index

expect_null(idx("cpu"))
expect_null(idx("mps"))
expect_null(idx(NULL))
expect_null(idx(NA_character_))
expect_null(idx(c("cuda", "cuda")))
expect_equal(idx("cuda"), 0L)
expect_equal(idx("cuda:0"), 0L)
expect_equal(idx("cuda:3"), 3L)
# A malformed index degrades to device 0 rather than erroring.
expect_equal(idx("cuda:notanumber"), 0L)

# torch_device objects still work, via as.character() -- which is pure
# formatting and creates no context.
expect_equal(idx(torch::torch_device("cuda:2")), 2L)
expect_null(idx(torch::torch_device("cpu")))

# ---- dtype sizing, no tensor allocated ----

eb <- whisper:::.gc_element_bytes
expect_equal(eb("float32", 0L), 4)
expect_equal(eb("float16", 0L), 2)
expect_equal(eb(torch::torch_float(), 0L), 4)
expect_equal(eb(torch::torch_float16(), 0L), 2)

# ---- fp16-broken detection is name-based, not torch-based ----

expect_true(whisper:::.fp16_broken_name("NVIDIA GeForce GTX 1660 Ti"))
expect_true(whisper:::.fp16_broken_name("NVIDIA GeForce GTX 1650"))
expect_false(whisper:::.fp16_broken_name("NVIDIA GeForce RTX 5060 Ti"))
expect_false(whisper:::.fp16_broken_name("NVIDIA A100-SXM4-40GB"))
expect_false(whisper:::.fp16_broken_name(NA_character_))

# ---- the tuner is a no-op off CUDA, and sets nothing ----

old <- options(torch.cuda_allocator_reserved_rate = NULL,
               torch.threshold_call_gc = NULL)
on.exit(options(old), add = TRUE)

expect_null(whisper_tune_gc("large-v3", device = "cpu"))
expect_null(getOption("torch.cuda_allocator_reserved_rate"))
expect_null(getOption("torch.threshold_call_gc"))

# ---- an explicit reserved rate is never overwritten ----

options(torch.cuda_allocator_reserved_rate = 0.55)
expect_null(whisper_tune_gc("large-v3", device = "cuda"))
expect_equal(getOption("torch.cuda_allocator_reserved_rate"), 0.55)
options(torch.cuda_allocator_reserved_rate = NULL)

# ---- the regression: it must not initialize CUDA ----
# Run in a subprocess so the check is real: once CUDA is initialized in a
# session there is no way to un-initialize it. cuda_is_available() is the
# probe the old implementation reached through parse_device("auto"); here
# nothing may call it before the options are set.

script <- tempfile(fileext = ".R")
on.exit(unlink(script), add = TRUE)
writeLines(c(
  'suppressMessages(library(whisper))',
  # Fail loudly if anything inside the tuner reaches for these.
  'trap <- function(...) stop("whisper_tune_gc initialized CUDA")',
  'assignInNamespace("cuda_is_available", trap, ns = "torch")',
  'assignInNamespace("cuda_current_device", trap, ns = "torch")',
  'r <- tryCatch({ whisper_tune_gc("tiny", device = "auto"); "ok" },',
  '              error = function(e) paste("FAILED:", conditionMessage(e)))',
  'cat(r, "\n")'
), script)

out <- suppressWarnings(system2(
  file.path(R.home("bin"), "Rscript"), c("--vanilla", shQuote(script)),
  stdout = TRUE, stderr = TRUE))
expect_false(any(grepl("initialized CUDA", out)))
