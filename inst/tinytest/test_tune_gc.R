# Tests for whisper_tune_gc() and its CUDA-free device/dtype resolution.
#
# The behaviour under test is a negative one: this function must NOT
# initialize CUDA. torch reads the allocator rates once, at CUDA init, so a
# tuner that probes first sets them after they were read -- a silent no-op
# that still prints a success message. Everything here therefore runs
# without torch being usable, which is also why these tests need no GPU.

# NOTE ON GUARDS. requireNamespace("torch") is NOT sufficient here: torch is
# an Import, so it installs on any check machine, while its Lantern runtime
# is a separate post-install download that a build host will not have.
# Constructing a torch object without Lantern errors. torch_is_installed()
# is the predicate that reports Lantern, which is why every other test file
# in this package uses it -- this one did not, and win-builder failed on
# exactly that.
#
# Most of what follows needs no torch at all, which is the point of the fix
# under test: device and dtype resolution happen without touching torch. So
# rather than skipping the file, only the few assertions that construct
# torch objects are gated.
have_torch <- requireNamespace("torch", quietly = TRUE) &&
  isTRUE(tryCatch(torch::torch_is_installed(), error = function(e) FALSE))

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

# ---- dtype sizing, no tensor allocated ----

eb <- whisper:::.gc_element_bytes
expect_equal(eb("float32", 0L), 4)
expect_equal(eb("float16", 0L), 2)

# ---- torch-object inputs (needs Lantern: constructing these loads it) ----

if (have_torch) {
  # torch_device objects work via as.character() -- pure formatting, which
  # creates no CUDA context.
  expect_equal(idx(torch::torch_device("cuda:2")), 2L)
  expect_null(idx(torch::torch_device("cpu")))
  expect_equal(eb(torch::torch_float(), 0L), 4)
  expect_equal(eb(torch::torch_float16(), 0L), 2)
}

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

# Assert the subprocess actually reached the end, not just that the trap
# string is absent: if it died early for any unrelated reason the grep
# below would find nothing and pass vacuously.
expect_true(any(grepl("^ok|^FAILED", out)))
expect_false(any(grepl("initialized CUDA", out)))
