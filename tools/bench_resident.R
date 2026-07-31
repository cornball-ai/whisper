#!/usr/bin/env r
# Benchmark repeated activation of a resident model.
#
# Usage: r tools/bench_resident.R [model] [n]
#        Rscript --vanilla tools/bench_resident.R [model] [n]
#
# Compares the path residency replaces -- a cold whisper_pipeline() load
# (disk -> CPU -> GPU) -- against resident_load() once followed by n
# activate/deactivate cycles (pinned RAM <-> GPU). GB/s is computed from
# the manifest's logical bytes; allocator numbers are printed as evidence
# only.

Sys.setenv(TORCH_VERIFY_LOAD = "FALSE")
suppressMessages({
  library(whisper)
  library(torch)
})

if (!exists("argv")) argv <- commandArgs(trailingOnly = TRUE)
model <- if (length(argv) >= 1) argv[1] else "medium"
n <- if (length(argv) >= 2) as.integer(argv[2]) else 10L

stopifnot(cuda_is_available())
alloc <- function() cuda_memory_stats()$allocated_bytes$all$current
gb <- function(b) b / 1024^3

cat("== bench_resident:", model, "x", n, "==\n")
smi <- tryCatch(system2("nvidia-smi",
  c("--query-gpu=name,memory.total", "--format=csv,noheader"),
  stdout = TRUE)[1], error = function(e) "nvidia-smi unavailable")
cat("gpu:", smi, "\n")
cat(sprintf("allocator baseline: %.3f GB\n", gb(alloc())))

# --- the path residency replaces ---
t_cold <- system.time(
  pipe <- whisper_pipeline(model, verbose = FALSE))[["elapsed"]]
peak_cold <- alloc()
cat(sprintf("cold whisper_pipeline(): %8.2f s   (allocator %.2f GB)\n",
  t_cold, gb(peak_cold)))
rm(pipe)
invisible(gc())
cuda_empty_cache()

# --- resident path ---
t_load <- system.time(
  res <- resident_load(model, verbose = FALSE))[["elapsed"]]
s <- resident_status(res)
bytes <- s$pinned_bytes
cat(sprintf("resident_load():         %8.2f s   (pins %.2f GB, dtype %s, sha256 %s...)\n",
  t_load, gb(bytes), s$dtype, substr(s$identity$weights_sha256, 1, 12)))

# Both deactivation modes. release = TRUE returns the allocator's blocks
# to the driver (other processes see the VRAM free) and the next
# activation re-acquires every block from the driver; release = FALSE
# keeps them pooled for the next model in THIS process to reuse. On a
# small card the difference is ~10x, so a single-process model host wants
# FALSE and the number to quote for it is the second row.
cycle <- function(release) {
  act <- numeric(n)
  deact <- numeric(n)
  peak <- 0
  for (i in seq_len(n)) {
    act[i] <- system.time(resident_activate(res))[["elapsed"]]
    peak <- max(peak, alloc())
    deact[i] <- system.time(
      resident_deactivate(res, release = release))[["elapsed"]]
  }
  list(act = act, deact = deact, peak = peak)
}

for (rel in c(TRUE, FALSE)) {
  r <- cycle(rel)
  cat(sprintf("release=%-5s activate   x%d: median %.3f s  min %.3f  max %.3f  (%.2f GB/s)\n",
    rel, n, median(r$act), min(r$act), max(r$act),
    gb(bytes) / median(r$act)))
  cat(sprintf("             deactivate x%d: median %.3f s  (allocator active %.2f GB, after %.3f GB)\n",
    n, median(r$deact), gb(r$peak), gb(alloc())))
  cat(sprintf("             speedup vs cold load: %.0fx (%.2f s -> %.3f s)\n",
    t_cold / median(r$act), t_cold, median(r$act)))
}

# sanity: the reactivated model still transcribes
resident_activate(res)
audio <- system.file("audio", "jfk.mp3", package = "whisper")
r <- resident_transcribe(res, audio, language = "en", timestamps = TRUE,
  verbose = FALSE)
cat("transcribe sanity:", substr(r$text, 1, 60), "\n")
resident_deactivate(res)
resident_unload(res)
