# In-process model residency: pinned host weights, disposable GPU copies.
#
# A resident model keeps its canonical weights as page-locked (pinned) CPU
# tensors for the life of the handle. Activation creates the GPU
# representation with a fast DMA copy from pinned memory; deactivation
# destroys only the GPU representation and repoints the module at the pinned
# host storage. Reactivation never touches the disk. This makes switching
# models on a small GPU a sub-second operation instead of a full reload.
#
# Mechanics rest on two torch behaviours verified against the installed
# version:
# - nn_module$to() REBINDS parameter/buffer objects (nn.R .apply wraps the
#   moved tensor in a fresh nn_parameter), so pinned host tensors held in
#   res$pinned survive activation -- and any tensor handle taken before a
#   transition is stale after it. All rebinding therefore resolves the
#   module's CURRENT tensors by name, every time.
# - Tensor$set_data() works across devices: a CUDA parameter can be
#   repointed directly at a pinned CPU tensor. That is the evict mechanism;
#   the orphaned CUDA storage is reclaimed by gc() + cuda_empty_cache().
#
# States: inactive -> activating -> active -> deactivating -> inactive.
# Failed transitions roll back to pinned host state; a rollback that cannot
# be verified leaves the handle "broken" (fail-closed: only status and
# unload work). "unloaded" is terminal.

#' Format a byte count for messages
#' @noRd
.fmt_bytes <- function(b) {
  if (is.null(b) || is.na(b) || b <= 0) {
    return("0 B")
  }
  u <- c("B", "KB", "MB", "GB", "TB")
  i <- min(floor(log(b, 1024)), 4)
  sprintf("%.2f %s", b / 1024^i, u[i + 1])
}

#' Enumerate the module's current named tensors (parameters + buffers)
#'
#' Always re-enumerated at the point of use: module$to() rebinds tensor
#' objects, so handles from load time go stale after every transition.
#' @noRd
.resident_tensors <- function(module) {
  c(module$named_parameters(), module$named_buffers())
}

#' Manifest of the model's tensors: name, kind, dtype, shape, logical bytes
#' @noRd
.resident_manifest <- function(module) {
  ts <- .resident_tensors(module)
  pn <- names(module$named_parameters())
  data.frame(
    name = names(ts),
    kind = ifelse(names(ts) %in% pn, "parameter", "buffer"),
    dtype = vapply(ts, function(t) as.character(t$dtype), character(1)),
    shape = vapply(ts, function(t) paste(dim(t), collapse = "x"),
      character(1)),
    bytes = vapply(ts, function(t) t$numel() * t$element_size(), numeric(1)),
    stringsAsFactors = FALSE, row.names = NULL
  )
}

#' Require exact manifest equality before a transition
#'
#' The current tensor set must match the manifest recorded at load: same
#' names, same dtypes, same shapes. Any mismatch aborts the transition.
#' Returns the current tensors, invisibly.
#' @noRd
.resident_check <- function(module, manifest) {
  cur <- .resident_tensors(module)
  if (!setequal(names(cur), manifest$name)) {
    stop("model tensors no longer match the residency manifest: ",
      "expected ", nrow(manifest), " names, found ", length(cur))
  }
  for (i in seq_len(nrow(manifest))) {
    t <- cur[[manifest$name[i]]]
    if (as.character(t$dtype) != manifest$dtype[i]) {
      stop("dtype changed for '", manifest$name[i], "': ",
        as.character(t$dtype), " vs manifest ", manifest$dtype[i])
    }
    if (paste(dim(t), collapse = "x") != manifest$shape[i]) {
      stop("shape changed for '", manifest$name[i], "'")
    }
  }
  invisible(cur)
}

#' TRUE when every manifest tensor sits on exactly the target device
#'
#' Exact equality, index included: a handle bound to cuda:0 must not pass
#' verification with tensors on cuda:1.
#' @noRd
.resident_on_target <- function(res) {
  ok <- tryCatch({
    cur <- .resident_tensors(res$pipe$model)
    if (!setequal(names(cur), res$manifest$name)) {
      FALSE
    } else {
      all(vapply(cur, function(t) t$device == res$target_device,
        logical(1)))
    }
  }, error = function(e) FALSE)
  isTRUE(ok)
}

#' Synchronize the handle's own GPU, never whichever device is current
#' @noRd
.resident_sync <- function(res) {
  torch::cuda_synchronize(device = res$target_device$index)
}

#' TRUE when every manifest tensor is back on CPU (i.e. pinned host state)
#'
#' Pinnedness itself is a load-time invariant: the tensors in res$pinned
#' were verified pinned once at load, a pinned allocation cannot silently
#' become pageable, and rollback rebinds to exactly those tensors by name.
#' So the transition check is: complete manifest coverage + every current
#' tensor on CPU. (Per-tensor is_pinned() is not used here because this
#' torch build requires the deprecated device argument, and libtorch emits
#' a stderr warning on every such call.)
#' @noRd
.resident_verify_pinned <- function(res) {
  ok <- tryCatch({
    cur <- .resident_tensors(res$pipe$model)
    if (!setequal(names(cur), res$manifest$name)) {
      return(FALSE)
    }
    all(vapply(cur, function(t) t$device$type == "cpu", logical(1)))
  }, error = function(e) FALSE)
  isTRUE(ok)
}

#' Rebind every current tensor to its pinned host copy and reclaim VRAM
#'
#' The shared workhorse of deactivation and of rollback after a failed
#' activation. Returns TRUE only when the restored pinned state has been
#' verified tensor-by-tensor. Synchronization and cache-release failures
#' make the rollback unprovable -- an asynchronous CUDA failure must not
#' let a handle return to "inactive" -- and the cause is appended to
#' last_error so it survives into the broken state.
#' @noRd
.resident_rollback <- function(res, release = TRUE) {
  cause <- NULL
  ok <- tryCatch({
    .resident_sync(res)
    cur <- .resident_tensors(res$pipe$model)
    if (!setequal(names(cur), names(res$pinned))) {
      cause <- "pinned set does not cover the current tensors"
      FALSE
    } else {
      torch::with_no_grad({
        for (nm in names(cur)) {
          cur[[nm]]$set_data(res$pinned[[nm]])
        }
      })
      gc()
      if (isTRUE(release)) {
        torch::cuda_empty_cache()
      }
      TRUE
    }
  }, error = function(e) {
    cause <<- conditionMessage(e)
    FALSE
  })
  if (!isTRUE(ok)) {
    res$last_error <- paste(
      c(res$last_error, paste0("rollback: ", cause %||% "unknown")),
      collapse = "; ")
    return(FALSE)
  }
  .resident_verify_pinned(res)
}

#' Default move seam: one non-blocking module move
#'
#' Kept as a seam (stored on the handle) so tests can inject a partial move
#' that fails midway and prove the rollback path.
#' @noRd
.resident_move <- function(res, device) {
  res$pipe$model$to(device = device, non_blocking = TRUE)
  invisible(NULL)
}

#' Refuse verbs on handles that are terminal or fail-closed
#' @noRd
.resident_guard <- function(res, verb) {
  if (!inherits(res, "whisper_resident")) {
    stop("not a whisper_resident handle")
  }
  if (identical(res$state, "unloaded")) {
    stop(verb, "(): handle has been unloaded")
  }
  if (identical(res$state, "broken")) {
    stop(verb, "(): handle is broken (", res$last_error %||% "unknown error",
      "); only resident_status() and resident_unload() are available")
  }
  invisible(res)
}

#' HF snapshot revision from an hfhub cache path, or NA
#'
#' Parses the path as given. normalizePath() must NOT be used here: the
#' hfhub layout stores snapshots/<revision>/model.safetensors as a symlink
#' into blobs/<hash>, so resolving symlinks erases the very path segment
#' this function exists to read.
#' @noRd
.snapshot_revision <- function(path) {
  parts <- strsplit(path.expand(path), "/", fixed = TRUE)[[1]]
  i <- which(parts == "snapshots")
  if (length(i) == 1 && length(parts) > i) parts[i + 1L] else NA_character_
}

#' Manifest bytes currently observed on the GPU, or NA when unknowable
#'
#' Honest accounting for broken partial transitions: a broken handle may
#' still hold some CUDA tensors, and the consumer budgeting VRAM needs the
#' observed number, not a stale one.
#' @noRd
.resident_gpu_bytes_observed <- function(res) {
  tryCatch({
    cur <- .resident_tensors(res$pipe$model)
    if (!setequal(names(cur), res$manifest$name)) {
      NA_real_
    } else {
      sum(vapply(cur, function(t) {
        if (t$device$type == "cuda") t$numel() * t$element_size() else 0
      }, numeric(1)))
    }
  }, error = function(e) NA_real_)
}

#' Load a Model as a Resident (Pinned Host Weights)
#'
#' Loads a Whisper model and retains its canonical weights as page-locked
#' (pinned) CPU tensors, so the GPU representation can be created and
#' destroyed repeatedly without reloading from disk.
#' [resident_activate()] copies the weights to the GPU (a fast DMA
#' transfer from pinned memory); [resident_deactivate()] frees the
#' GPU copy and repoints the model at the pinned host storage. The handle
#' starts inactive.
#'
#' The dtype is resolved against the *target* device with the same rules as
#' [whisper_pipeline()]: float16 on CUDA, except GPUs with broken
#' fp16 (GTX 16-series), which get float32. The pinned copies are stored at
#' the resolved dtype, so activation moves exactly the bytes inference needs.
#'
#' @param model Model name: "tiny", "base", "small", "medium", "large-v3"
#' @param device Target CUDA device for activation (default "cuda").
#'   Residency requires CUDA; pinned host memory exists to feed it. A bare
#'   "cuda" is resolved to the current device's explicit index at load
#'   time, and the handle stays bound to that exact device (`"cuda:N"`)
#'   for every later transition and synchronize.
#' @param dtype "auto" (default), "float16", or "float32"; resolved against
#'   `device`.
#' @param download Download the model if not cached (default TRUE).
#' @param verbose Print progress messages.
#'
#' @return A `whisper_resident` handle (an environment). Fields of
#'   interest via [resident_status()]: state, byte counts, and a
#'   content identity (weights sha256, HF repo and snapshot revision,
#'   resolved dtype).
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed() && torch::cuda_is_available() &&
#'   model_exists("tiny")) {
#'   res <- resident_load("tiny")
#'   resident_activate(res)
#'   audio <- system.file("audio", "jfk.mp3", package = "whisper")
#'   resident_transcribe(res, audio, timestamps = TRUE)
#'   resident_deactivate(res) # VRAM freed, weights stay pinned in RAM
#'   resident_activate(res) # fast: DMA copy, no disk
#'   resident_unload(res)
#' }
#' }
#' @export
resident_load <- function(
  model = "tiny",
  device = "cuda",
  dtype = "auto",
  download = TRUE,
  verbose = TRUE
) {
  target <- parse_device(device)
  if (target$type != "cuda") {
    stop("resident_load() requires a CUDA target device")
  }
  if (!torch::cuda_is_available()) {
    stop("CUDA is not available")
  }
  # Bind the handle to one explicit GPU. A bare "cuda" resolves to the
  # current device NOW; every later synchronize and device verification
  # uses this index, so the handle cannot drift to whichever GPU happens
  # to be current at transition time.
  if (is.null(target$index)) {
    target <- torch::torch_device(
      paste0("cuda:", torch::cuda_current_device()))
  }
  resolved_dtype <- parse_dtype(dtype, target)

  # Build on CPU at the TARGET dtype; the device move is what
  # resident_activate() exists for.
  whisper <- load_whisper_model(model, device = torch::torch_device("cpu"),
    dtype = resolved_dtype, download = download, verbose = verbose)
  tokenizer <- whisper_tokenizer(model)
  config <- whisper_config(model)

  manifest <- .resident_manifest(whisper)

  if (verbose) {
    message("Pinning ", nrow(manifest), " tensors (",
      .fmt_bytes(sum(manifest$bytes)), ") in host memory")
  }
  pinned <- list()
  torch::with_no_grad({
    ts <- .resident_tensors(whisper)
    for (nm in names(ts)) {
      t <- ts[[nm]]
      pinned[[nm]] <- t$detach()$pin_memory(target)
      t$set_data(pinned[[nm]])
    }
  })
  # The pageable originals lost their last reference in set_data; drop them
  # so the pinned copies are the model's only host representation.
  gc()

  # Verify pinnedness once, here, where it is established. This is the only
  # place is_pinned() runs outside the tests: the R binding requires the
  # deprecated device argument, so each call draws a libtorch stderr
  # warning -- a bounded burst at load, not noise on every transition.
  for (nm in names(pinned)) {
    if (!pinned[[nm]]$is_pinned(target)) {
      stop("pin_memory() did not produce a pinned tensor for '", nm, "'")
    }
  }

  weights_path <- get_weights_path(model)
  if (verbose) message("Hashing weights (sha256)")
  identity <- list(
    model = model,
    repo = config$hf_repo,
    revision = .snapshot_revision(weights_path),
    weights_sha256 = unname(tools::sha256sum(weights_path)),
    weights_bytes = unname(file.size(weights_path)),
    dtype = as.character(resolved_dtype)
  )

  res <- new.env(parent = emptyenv())
  res$pipe <- list(model = whisper, tokenizer = tokenizer, config = config,
    device = target, dtype = resolved_dtype)
  res$pinned <- pinned
  res$manifest <- manifest
  res$state <- "inactive"
  res$in_flight <- FALSE
  res$identity <- identity
  res$weights_path <- weights_path
  res$target_device <- target
  res$pinned_bytes <- sum(manifest$bytes)
  res$gpu_bytes <- 0
  res$move <- .resident_move
  res$last_error <- NULL
  class(res) <- "whisper_resident"
  res
}

#' Activate a Resident Model (Create the GPU Representation)
#'
#' Copies the pinned host weights to the target device in one synchronous
#' pass (non-blocking per-tensor copies, then a device synchronize). The
#' transition is transactional: on any failure -- including a partial move
#' after an out-of-memory error -- every tensor is rebound to its pinned
#' host copy, the CUDA allocation is released, and the handle returns to
#' "inactive". A rollback that cannot be verified leaves the handle
#' "broken", where only [resident_status()] and
#' [resident_unload()] operate.
#'
#' @param res A `whisper_resident` handle from [resident_load()].
#' @return The handle, invisibly. No-op when already active.
#' @export
resident_activate <- function(res) {
  .resident_guard(res, "resident_activate")
  if (identical(res$state, "active")) {
    return(invisible(res))
  }
  if (!identical(res$state, "inactive")) {
    stop("cannot activate from state '", res$state, "'")
  }
  # Pre-move manifest check: abort while still inactive, nothing moved yet.
  .resident_check(res$pipe$model, res$manifest)

  res$state <- "activating"
  err <- NULL
  ok <- tryCatch({
    res$move(res, res$target_device)
    .resident_sync(res)
    TRUE
  }, error = function(e) {
    err <<- conditionMessage(e)
    FALSE
  })

  if (ok && .resident_on_target(res)) {
    res$state <- "active"
    res$gpu_bytes <- res$pinned_bytes
    return(invisible(res))
  }
  if (is.null(err)) err <- "device verification failed after move"
  res$last_error <- err

  if (.resident_rollback(res)) {
    res$state <- "inactive"
    res$gpu_bytes <- 0
    stop("resident_activate() failed (rolled back to pinned host state): ",
      err)
  }
  res$state <- "broken"
  res$gpu_bytes <- .resident_gpu_bytes_observed(res)
  stop("resident_activate() failed and rollback could not be verified; ",
    "handle is broken: ", res$last_error)
}

#' Deactivate a Resident Model (Destroy the GPU Representation)
#'
#' Rebinds every tensor to its pinned host copy, releases the orphaned CUDA
#' storage, and verifies the restored state tensor-by-tensor. The pinned
#' weights remain in host memory, so a later [resident_activate()]
#' is a DMA copy, not a disk reload. Refuses while a transcription is in
#' flight.
#'
#' `release` decides who gets the freed VRAM, and it is worth about an
#' order of magnitude. With `release = TRUE` (the default) the CUDA
#' caching allocator hands its blocks back to the driver, so the memory is
#' visible as free to other processes -- but the next activation must
#' re-acquire every block from the driver, which on a small card measured
#' ~9 ms per tensor (medium fp32, 2.85 GB across 948 tensors: 9.2 s,
#' 0.31 GB/s). With `release = FALSE` the blocks stay in this process's
#' pool for the next model to reuse, and the same activation took 0.86 s
#' (3.29 GB/s), matching raw pinned-DMA bandwidth on that card. Use
#' `FALSE` when one process hosts every model and switches between
#' them; use the default when a different process needs the card.
#'
#' @param res A `whisper_resident` handle.
#' @param release Return the allocator's blocks to the driver (default
#'   TRUE). FALSE keeps them pooled for a fast next activation; the
#'   weights are freed either way, and `gpu_bytes` goes to zero in both
#'   cases (the retained pool is process overhead, attributable to no
#'   model).
#' @return The handle, invisibly. No-op when already inactive.
#' @export
resident_deactivate <- function(res, release = TRUE) {
  .resident_guard(res, "resident_deactivate")
  if (isTRUE(res$in_flight)) {
    stop("resident_deactivate(): a transcription is in flight")
  }
  if (identical(res$state, "inactive")) {
    return(invisible(res))
  }
  if (!identical(res$state, "active")) {
    stop("cannot deactivate from state '", res$state, "'")
  }
  res$state <- "deactivating"
  if (.resident_rollback(res, release = release)) {
    res$state <- "inactive"
    res$gpu_bytes <- 0
    return(invisible(res))
  }
  res$state <- "broken"
  res$gpu_bytes <- .resident_gpu_bytes_observed(res)
  stop("resident_deactivate() could not verify the pinned host state; ",
    "handle is broken", if (is.null(res$last_error)) "" else
      paste0(": ", res$last_error))
}

#' Transcribe with a Resident Model
#'
#' Runs [transcribe()] through the resident handle's pipeline.
#' Requires the model to be active; marks the handle in-flight for the
#' duration so a concurrent deactivation is refused.
#'
#' @param res A `whisper_resident` handle, currently active.
#' @param file Path to audio file (WAV, MP3, MP4, ...).
#' @param ... Passed to the pipeline transcriber: `language`,
#'   `task`, `timestamps`, `word_timestamps`,
#'   `beam_size`, `temperatures`, ... (see [transcribe()]).
#' @return The transcription result, same shape as [transcribe()].
#' @export
resident_transcribe <- function(res, file, ...) {
  .resident_guard(res, "resident_transcribe")
  if (!identical(res$state, "active")) {
    stop("resident_transcribe(): model is not active; ",
      "call resident_activate() first")
  }
  if (isTRUE(res$in_flight)) {
    stop("resident_transcribe(): another transcription is in flight")
  }
  res$in_flight <- TRUE
  on.exit(res$in_flight <- FALSE, add = TRUE)
  pipeline_transcribe(res$pipe, file, ...)
}

#' Status of a Resident Model
#'
#' Callable in every state, including "broken" and "unloaded".
#'
#' @param res A `whisper_resident` handle.
#' @return A list: model, state, in_flight, device (the exact bound device,
#'   e.g. `"cuda:0"`), dtype, pinned_bytes, gpu_bytes (logical sums over
#'   the tensor manifest, not allocator statistics; in the "broken" state
#'   this is the observed on-GPU manifest bytes, or NA when unknowable),
#'   identity (weights sha256, HF repo/revision, resolved dtype), path
#'   (diagnostic), last_error.
#' @export
resident_status <- function(res) {
  if (!inherits(res, "whisper_resident")) {
    stop("not a whisper_resident handle")
  }
  list(
    model = res$identity$model,
    state = res$state,
    in_flight = isTRUE(res$in_flight),
    device = if (is.null(res$target_device)) NA_character_ else
      as.character(res$target_device),
    dtype = res$identity$dtype,
    pinned_bytes = if (identical(res$state, "unloaded")) 0 else
      res$pinned_bytes,
    gpu_bytes = res$gpu_bytes,
    identity = res$identity,
    path = res$weights_path,
    last_error = res$last_error
  )
}

#' Unload a Resident Model (Drop Pinned and GPU State)
#'
#' Releases both representations: any GPU tensors and the pinned host
#' copies. Permitted from every state except mid-flight -- including
#' "broken", where it is the recovery path (dropping the module and the
#' pinned list releases the storage that could not be verified). The handle
#' becomes "unloaded", which is terminal.
#'
#' @param res A `whisper_resident` handle.
#' @return The handle, invisibly.
#' @export
resident_unload <- function(res) {
  if (!inherits(res, "whisper_resident")) {
    stop("not a whisper_resident handle")
  }
  if (isTRUE(res$in_flight)) {
    stop("resident_unload(): a transcription is in flight")
  }
  if (identical(res$state, "unloaded")) {
    return(invisible(res))
  }
  if (identical(res$state, "active")) {
    try(.resident_rollback(res), silent = TRUE)
  }
  res$pinned <- NULL
  res$pipe <- NULL
  res$gpu_bytes <- 0
  res$pinned_bytes <- 0
  res$state <- "unloaded"
  gc()
  try(torch::cuda_empty_cache(), silent = TRUE)
  invisible(res)
}

#' Print a Resident Handle
#'
#' @param x A `whisper_resident` handle.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.whisper_resident <- function(x, ...) {
  cat(sprintf("<whisper_resident: %s [%s] pinned %s>\n",
    if (is.null(x$identity)) "?" else x$identity$model,
    x$state, .fmt_bytes(x$pinned_bytes)))
  invisible(x)
}
