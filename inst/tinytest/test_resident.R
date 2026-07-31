# Tests for in-process model residency (R/resident.R).
#
# CPU-safe section runs everywhere. The CUDA section needs a GPU and a
# downloaded model, so it is gated on cuda_is_available() && at_home().

if (!requireNamespace("torch", quietly = TRUE) ||
  !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

# ---- byte formatting ----
fmt <- whisper:::.fmt_bytes
expect_equal(fmt(0), "0 B")
expect_equal(fmt(1024), "1.00 KB")
expect_equal(fmt(3.1 * 1024^3), "3.10 GB")

# ---- closed-behavior matrix on fabricated handles (no GPU needed) ----
# Verbs must hit the state guards before touching any torch machinery, so a
# minimal fabricated handle exercises them.

fab <- function(state, in_flight = FALSE) {
  res <- new.env(parent = emptyenv())
  res$state <- state
  res$in_flight <- in_flight
  res$identity <- list(model = "fake", dtype = "Float")
  res$pinned_bytes <- 0
  res$gpu_bytes <- 0
  res$last_error <- "synthetic"
  class(res) <- "whisper_resident"
  res
}

# unloaded: everything except status/unload refuses
u <- fab("unloaded")
expect_error(resident_activate(u), pattern = "unloaded")
expect_error(resident_deactivate(u), pattern = "unloaded")
expect_error(resident_transcribe(u, "x.mp3"), pattern = "unloaded")
expect_equal(resident_status(u)$state, "unloaded")
expect_silent(resident_unload(u)) # no-op, stays unloaded
expect_equal(u$state, "unloaded")

# broken: fail-closed, status reports, unload recovers
b <- fab("broken")
expect_error(resident_activate(b), pattern = "broken")
expect_error(resident_deactivate(b), pattern = "broken")
expect_error(resident_transcribe(b, "x.mp3"), pattern = "broken")
expect_equal(resident_status(b)$state, "broken")
expect_equal(resident_status(b)$last_error, "synthetic")
expect_silent(resident_unload(b))
expect_equal(b$state, "unloaded")

# inactive: transcribe refused with actionable message
i <- fab("inactive")
expect_error(resident_transcribe(i, "x.mp3"), pattern = "resident_activate")

# active + in_flight: deactivate/unload/second transcribe refused
a <- fab("active", in_flight = TRUE)
expect_error(resident_deactivate(a), pattern = "in flight")
expect_error(resident_unload(a), pattern = "in flight")
expect_error(resident_transcribe(a, "x.mp3"), pattern = "in flight")

# non-handles refused everywhere
expect_error(resident_status(list()), pattern = "whisper_resident")
expect_error(resident_activate(list()), pattern = "whisper_resident")

# ---- manifest equality machinery (CPU torch, no GPU needed) ----

m <- torch::nn_linear(4, 4)
man <- whisper:::.resident_manifest(m)
expect_equal(sort(man$name), c("bias", "weight"))
expect_true(all(man$kind == "parameter"))
expect_equal(man$bytes[man$name == "weight"], 4 * 4 * 4)

# named_buffers ride along in the manifest
bnm <- torch::nn_batch_norm1d(4)
manb <- whisper:::.resident_manifest(bnm)
expect_true("running_mean" %in% manb$name[manb$kind == "buffer"])

# exact-equality check: pass, then fail on a mutated manifest
expect_silent(whisper:::.resident_check(m, man))
man_bad <- man
man_bad$name[1] <- "nonexistent"
expect_error(whisper:::.resident_check(m, man_bad), pattern = "manifest")
man_bad2 <- man
man_bad2$shape[man_bad2$name == "weight"] <- "9x9"
expect_error(whisper:::.resident_check(m, man_bad2), pattern = "shape")

# ---- CUDA section ----

if (!torch::cuda_is_available()) exit_file("CUDA not available")
if (!at_home()) exit_file("CUDA residency tests only run at home")
if (!whisper::model_exists("tiny")) exit_file("model 'tiny' not downloaded")

dev <- torch::torch_device("cuda")
audio <- system.file("audio", "jfk.mp3", package = "whisper")
alloc <- function() {
  torch::cuda_memory_stats()$allocated_bytes$all$current
}

res <- resident_load("tiny", verbose = FALSE)

# Exact name coverage: pinned == manifest == current params+buffers,
# and every retained tensor is pinned (both real buffers included).
cur <- whisper:::.resident_tensors(res$pipe$model)
expect_true(setequal(names(res$pinned), res$manifest$name))
expect_true(setequal(names(cur), res$manifest$name))
expect_true("encoder.positional_embedding" %in% names(res$pinned))
expect_true("decoder.mask" %in% names(res$pinned))
expect_true(all(vapply(names(res$pinned), function(nm) {
  res$pinned[[nm]]$is_pinned(dev)
}, logical(1))))

# Byte counts are logical sums over the manifest
expect_equal(res$pinned_bytes, sum(res$manifest$bytes))
expect_equal(resident_status(res)$gpu_bytes, 0)

# Identity: content digest + resolved repo/revision. The revision must be
# an actual snapshot sha, not NA -- normalizePath() used to resolve the
# hfhub snapshots/<rev> symlink into blobs/ and erase it.
st <- resident_status(res)
expect_equal(nchar(st$identity$weights_sha256), 64)
expect_equal(st$identity$repo, "openai/whisper-tiny")
expect_false(is.na(st$identity$revision))
expect_true(grepl("^[0-9a-f]{40}$", st$identity$revision))

# The handle is bound to an exact device, index included
expect_true(grepl("^cuda:[0-9]+$", st$device))

# ---- full cycle: deterministic settings, text + tensor equivalence ----
base_vram <- alloc()

# Fixed encoder input, created once and reused: n_mels from the conv1
# manifest shape ("<out>x<n_mels>x3"), padded length 3000 as the encoder
# expects. Comparison uses a tolerance, not bitwise identity: cuDNN
# algorithm selection between runs is not contractually stable.
nmels <- as.integer(strsplit(
  res$manifest$shape[res$manifest$name == "encoder.conv1.weight"],
  "x")[[1]][2])
x_cpu <- torch::torch_randn(1, nmels, 3000)
enc_out <- function(r) {
  torch::with_no_grad({
    x <- x_cpu$to(device = dev, dtype = r$pipe$dtype)
    r$pipe$model$encoder(x)$cpu()$to(dtype = torch::torch_float())
  })
}

det <- list(language = "en", timestamps = TRUE, beam_size = 1L,
  temperatures = 0, verbose = FALSE)

resident_activate(res)
expect_equal(res$state, "active")
expect_equal(resident_status(res)$gpu_bytes, res$pinned_bytes)

r1 <- do.call(resident_transcribe, c(list(res, audio), det))
e1 <- enc_out(res)

resident_deactivate(res)
expect_equal(res$state, "inactive")
expect_equal(resident_status(res)$gpu_bytes, 0)
# VRAM reclaimed (allocator evidence, never part of gpu_bytes). 16 MB slack
# for allocator rounding; tiny's weights alone are ~75 MB in fp16.
expect_true(alloc() <= base_vram + 16 * 1024^2)

resident_activate(res)
r2 <- do.call(resident_transcribe, c(list(res, audio), det))
e2 <- enc_out(res)

# text-level and tensor-level equivalence across cycles (the transcribe
# result does not expose token ids; the encoder-output comparison below is
# the lower-level check)
expect_identical(r1$text, r2$text)
expect_equal(r1$segments$text, r2$segments$text)
expect_true((e1 - e2)$abs()$max()$item() <= 1e-4)

# equivalence vs a fresh non-resident transcribe
rf <- do.call(whisper::transcribe, c(list(audio, model = "tiny"), det))
expect_identical(r1$text, rf$text)

# subtitle-interop shape rides along
expect_true(inherits(r1, "whisper_transcription"))

# ---- release = FALSE frees the weights but keeps the allocator pool ----
# Both modes must zero gpu_bytes and return every tensor to pinned host
# memory; they differ only in whether the blocks go back to the driver.
# Retaining them is what makes switching fast in a single-process host.
resident_activate(res)
resident_deactivate(res, release = FALSE)
expect_equal(res$state, "inactive")
expect_equal(resident_status(res)$gpu_bytes, 0)
expect_true(whisper:::.resident_verify_pinned(res))
# allocated (per-tensor) drops; reserved (pooled) is allowed to stay
expect_true(alloc() <= base_vram + 16 * 1024^2)
expect_true(torch::cuda_memory_stats()$reserved_bytes$all$current >=
  torch::cuda_memory_stats()$allocated_bytes$all$current)
# and the handle still works afterwards
resident_activate(res)
r_keep <- do.call(resident_transcribe, c(list(res, audio), det))
expect_identical(r1$text, r_keep$text)
resident_deactivate(res)

# ---- injected mid-activation failure: rollback proven ----
resident_deactivate(res)
expect_equal(res$state, "inactive")

real_move <- res$move
res$move <- function(res, device) {
  ts <- whisper:::.resident_tensors(res$pipe$model)
  nms <- names(ts)[seq_len(5)]
  torch::with_no_grad({
    for (nm in nms) ts[[nm]]$set_data(ts[[nm]]$to(device = device))
  })
  stop("injected mid-move failure")
}
expect_error(resident_activate(res), pattern = "rolled back")
expect_equal(res$state, "inactive")
# rollback proof: every manifest tensor back on CPU, VRAM reclaimed
expect_true(whisper:::.resident_verify_pinned(res))
expect_true(alloc() <= base_vram + 16 * 1024^2)
res$move <- real_move

# recovery: the handle still activates and transcribes correctly
resident_activate(res)
r3 <- do.call(resident_transcribe, c(list(res, audio), det))
expect_identical(r1$text, r3$text)
resident_deactivate(res)

# ---- two resident handles alternating on one GPU ----
if (whisper::model_exists("base")) {
  res_b <- resident_load("base", verbose = FALSE)
  for (round in 1:2) {
    resident_activate(res)
    ra <- do.call(resident_transcribe, c(list(res, audio), det))
    expect_identical(ra$text, r1$text)
    resident_deactivate(res)

    resident_activate(res_b)
    rb <- do.call(resident_transcribe, c(list(res_b, audio), det))
    expect_true(nchar(rb$text) > 0)
    resident_deactivate(res_b)
  }
  resident_unload(res_b)
  expect_equal(res_b$state, "unloaded")
}

# ---- unprovable rollback -> broken, and unload as the recovery path ----
# Corrupt the pinned set so a failed activation cannot prove its rollback
# (name coverage check fails), AND move part of the model first, so the
# broken handle genuinely holds CUDA tensors. gpu_bytes must then report
# the observed on-GPU manifest bytes, not a stale zero.
expect_equal(res$state, "inactive")
res$pinned[["decoder.mask"]] <- NULL
res$move <- function(res, device) {
  ts <- whisper:::.resident_tensors(res$pipe$model)
  nms <- names(ts)[seq_len(5)]
  torch::with_no_grad({
    for (nm in nms) ts[[nm]]$set_data(ts[[nm]]$to(device = device))
  })
  stop("injected failure, corrupted pinned")
}
expect_error(resident_activate(res), pattern = "broken")
expect_equal(res$state, "broken")
# the rollback cause survives into last_error
expect_true(grepl("rollback:", resident_status(res)$last_error))
# honest accounting: the manifest is still enumerable here, so gpu_bytes
# must be the observed sum of the 5 moved tensors -- not zero, not all,
# and not NA (NA is reserved for the genuinely unknowable case)
gb_broken <- resident_status(res)$gpu_bytes
expect_true(gb_broken > 0 && gb_broken < res$pinned_bytes)
expect_error(resident_transcribe(res, audio), pattern = "broken")
expect_error(resident_deactivate(res), pattern = "broken")
expect_equal(resident_status(res)$state, "broken")

resident_unload(res)
expect_equal(res$state, "unloaded")
expect_error(resident_activate(res), pattern = "unloaded")
expect_equal(resident_status(res)$pinned_bytes, 0)
# unload released the stranded CUDA tensors too
expect_true(alloc() <= base_vram + 16 * 1024^2)
