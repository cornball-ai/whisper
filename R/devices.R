#' Device and Dtype Management
#'
#' Utilities for managing torch devices and data types.

#' Get Default Device
#'
#' Returns CUDA device if available, otherwise CPU.
#'
#' @return torch device object
#' @export
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'   device <- whisper_device()
#'   device$type
#' }
#' }
whisper_device <- function() {

  if (torch::cuda_is_available()) {
    torch::torch_device("cuda")
  } else {
    torch::torch_device("cpu")
  }
}

#' Get Default Dtype
#'
#' Returns float16 on CUDA, float32 on CPU. Exception: the GTX 16-series
#' (TU116/TU117, e.g. GTX 1660/1650) computes float16 incorrectly and
#' produces NaN output, so float32 is used on those cards. Pass an explicit
#' \code{dtype = "float16"} to override.
#'
#' @param device torch device
#' @return torch dtype
#' @export
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'   dtype <- whisper_dtype()
#'   dtype
#' }
#' }
whisper_dtype <- function(device = whisper_device()) {
  # float16 on CUDA, float32 on CPU - but fall back to float32 on GPUs with
  # broken fp16 (see .fp16_broken_gpu).
  if (device$type == "cuda") {
    if (.fp16_broken_gpu(device)) {
      .fp16_warn_once()
      torch::torch_float()
    } else {
      torch::torch_float16()
    }
  } else {
    torch::torch_float()
  }
}

# The GTX 16-series (TU116/TU117: GTX 1630/1650/1660 and their Ti/Super
# variants) computes fp16 incorrectly - it produces NaN, surfacing as
# repeated "!" tokens in transcription, the same hardware quirk behind the
# "GTX 1660 black image" bug in other fp16 inference stacks. Detect by GPU
# name and fall back to fp32. CUDA-gated and tryCatch-guarded, so it never
# runs (or errors) on a CRAN check machine.
# GPU name for a device index, via nvidia-smi. Deliberately NOT via torch:
# callers include the GC tuner, which must not create a CUDA context.
.gpu_name <- function(idx = 0L) {
  name <- tryCatch(
    system2("nvidia-smi", c("--query-gpu=name", "--format=csv,noheader",
      paste0("--id=", idx)), stdout = TRUE)[1],
    error = function(e) NA_character_)
  if (length(name) != 1L || is.na(name) || !nzchar(name)) NA_character_ else name
}

.fp16_broken_name <- function(name) {
  !is.na(name) && grepl("GTX\\s*16", name, ignore.case = TRUE)
}

.fp16_broken_gpu <- function(device = whisper_device()) {
  if (!inherits(device, "torch_device") || device$type != "cuda") {
    return(FALSE)
  }
  idx <- device$index
  if (is.null(idx) || is.na(idx)) {
    idx <- 0L
  }
  .fp16_broken_name(.gpu_name(idx))
}

# Resolve a device argument to a CUDA index WITHOUT touching torch.
#
# whisper_tune_gc() cannot use parse_device(): "auto" routes through
# whisper_device() -> cuda_is_available(), which initializes CUDA. torch
# reads the allocator rates once, at that init, so probing first makes the
# tuner a silent no-op -- it sets the options after they have been read and
# still reports success. Everything here is strings and nvidia-smi.
#
# Returns an integer index for CUDA, or NULL for "no CUDA target".
.gc_device_index <- function(device) {
  if (inherits(device, "torch_device")) {
    device <- as.character(device)
  }
  if (!is.character(device) || length(device) != 1L || is.na(device)) {
    return(NULL)
  }
  if (identical(device, "auto")) {
    # A GPU that nvidia-smi can see is the best available signal without
    # creating a context. If there is none, there is nothing to tune.
    if (is.na(.gpu_name(0L))) {
      return(NULL)
    }
    return(0L)
  }
  if (!grepl("^cuda", device)) {
    return(NULL)
  }
  idx <- suppressWarnings(as.integer(sub("^cuda:?", "", device)))
  if (length(idx) != 1L || is.na(idx)) 0L else idx
}

# Bytes per parameter for a dtype argument, without allocating a tensor.
.gc_element_bytes <- function(dtype, idx) {
  if (identical(dtype, "float32")) {
    return(4)
  }
  if (identical(dtype, "float16")) {
    return(2)
  }
  if (inherits(dtype, "torch_dtype")) {
    return(if (grepl("Half", as.character(dtype), fixed = TRUE)) 2 else 4)
  }
  # "auto": fp16 on CUDA, except the GTX 16-series, which computes it wrong.
  if (.fp16_broken_name(.gpu_name(idx))) 4 else 2
}

# One-time message per session when falling back to fp32.
.whisper_dtype_env <- new.env(parent = emptyenv())
.fp16_warn_once <- function() {
  if (is.null(.whisper_dtype_env$fp16)) {
    message("whisper: GTX 16-series GPU detected; using float32 ",
      "(fp16 produces NaN on these cards). ",
      "Override with dtype = \"float16\".")
    .whisper_dtype_env$fp16 <- TRUE
  }
}

#' Parse Device Argument
#'
#' @param device Character or torch device. "auto" uses GPU if available.
#' @return torch device object
parse_device <- function(device = "auto") {

  if (is.character(device)) {
    if (device == "auto") {
      whisper_device()
    } else {
      torch::torch_device(device)
    }
  } else {
    device
  }
}

#' Parse Dtype Argument
#'
#' @param dtype Character or torch dtype. "auto" uses float16 on GPU, float32 on CPU.
#' @param device torch device (used for auto selection)
#' @return torch dtype
parse_dtype <- function(
  dtype = "auto",
  device = whisper_device()
) {
  if (is.character(dtype)) {
    if (dtype == "auto") {
      whisper_dtype(device)
    } else if (dtype == "float16") {
      torch::torch_float16()
    } else if (dtype == "float32") {
      torch::torch_float()
    } else {
      stop("Unknown dtype: ", dtype, ". Supported: auto, float16, float32")
    }
  } else {
    dtype
  }
}

#' Tune torch's CUDA garbage collection for whisper inference
#'
#' Opt-in performance helper. torch's CUDA allocator invokes R's \code{gc()}
#' on nearly every allocation once a loaded model occupies more than 20\% of
#' GPU memory (its default \code{torch.cuda_allocator_reserved_rate} floor),
#' which can dominate inference time for the larger whisper models. This raises
#' the floor to the model's footprint as a fraction of VRAM (clamped to at
#' least 0.2, so models already under the default are unaffected) and lifts
#' \code{torch.threshold_call_gc} off its 4 GB default.
#'
#' @details
#' Call this \emph{before} \code{\link{load_whisper_model}}: torch reads the
#' allocator rates once, at lazy CUDA initialization. It is a no-op on
#' non-CUDA devices and only sets an option that is not already set, so an
#' explicit \code{options(torch.cuda_allocator_reserved_rate = ...)} always
#' wins.
#'
#' \strong{Side effect:} it sets session-global \code{torch.*} options that
#' persist after the call - deliberately, since torch reads them later. The
#' package never calls this for you; you invoke it.
#'
#' \strong{Call it before anything initializes CUDA.} torch reads the
#' allocator rates exactly once, at CUDA init, so options set afterwards
#' have no effect for the rest of the session. This function is careful not
#' to initialize CUDA itself - it resolves the device and the dtype size
#' from strings and \code{nvidia-smi} rather than from torch - but it
#' cannot undo an init that already happened. In a process that hosts
#' several models, call it first, before any of them load.
#'
#' For several models resident on one GPU in the \emph{same} R process, pass
#' their combined size via \code{footprint_gb} so the single shared floor
#' covers all of them.
#'
#' @param model Whisper model name, used to estimate the footprint when
#'   \code{footprint_gb} is NULL.
#' @param device Device, as accepted by \code{\link{load_whisper_model}}.
#' @param dtype Compute dtype; determines bytes per parameter.
#' @param footprint_gb Optional explicit footprint in GB, overriding the
#'   per-model estimate (use for combined multi-model workloads).
#' @return The reserved-rate that was set (invisibly), or NULL when nothing
#'   was set (non-CUDA device, or the option was already set).
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'   # No-op off CUDA; returns NULL.
#'   whisper_tune_gc("large-v3", device = "cpu")
#' }
#' }
#' \dontrun{
#' # On a GPU, call before loading so torch picks up the rate at CUDA init:
#' whisper_tune_gc("large-v3", device = "cuda")
#' model <- load_whisper_model("large-v3", device = "cuda")
#' }
#' @export
whisper_tune_gc <- function(model = "large-v3", device = "auto",
                            dtype = "auto", footprint_gb = NULL) {
  # Nothing in this function may touch torch. It exists to set options that
  # torch reads exactly once, at CUDA init, so any call that initializes
  # CUDA first -- parse_device("auto"), parse_dtype(), allocating a tensor
  # to measure its element size -- makes the whole thing a silent no-op:
  # the options get set after they were read, and the message still claims
  # success. Device resolution and dtype sizing therefore go through
  # .gc_device_index()/.gc_element_bytes(), which use strings and
  # nvidia-smi only.
  idx <- .gc_device_index(device)
  if (is.null(idx)) {
    return(invisible(NULL))
  }
  if (is.null(getOption("torch.threshold_call_gc"))) {
    options(torch.threshold_call_gc = 16000)
  }
  if (!is.null(getOption("torch.cuda_allocator_reserved_rate"))) {
    return(invisible(NULL))
  }
  if (is.null(footprint_gb)) {
    el_bytes <- .gc_element_bytes(dtype, idx)
    footprint_gb <- .whisper_param_count(model) * el_bytes / 1e9
  }
  total_gb <- tryCatch(
    as.numeric(system2("nvidia-smi",
      c("--query-gpu=memory.total", "--format=csv,noheader,nounits",
        paste0("--id=", idx)), stdout = TRUE)[1]) / 1024,
    error = function(e) NA_real_)
  if (is.na(total_gb) || total_gb <= 0 ||
      is.na(footprint_gb) || footprint_gb <= 0) {
    return(invisible(NULL))
  }
  rate <- min(0.92, max(0.20, footprint_gb / total_gb))
  options(torch.cuda_allocator_reserved_rate = rate)
  message(sprintf(paste0("whisper: torch.cuda_allocator_reserved_rate = %.2f,",
    " threshold_call_gc = %d MB (%.1f GB model, %.0f GB VRAM)"),
    rate, getOption("torch.threshold_call_gc"), footprint_gb, total_gb))
  invisible(rate)
}

# Approximate parameter count per whisper model, for the GC footprint estimate.
.whisper_param_count <- function(model) {
  m <- sub("\\.en$", "", tolower(model))
  switch(m,
    tiny = 39e6,
    base = 74e6,
    small = 244e6,
    medium = 769e6,
    "large-v3-turbo" = 809e6,
    "large" = 1550e6,
    "large-v1" = 1550e6,
    "large-v2" = 1550e6,
    "large-v3" = 1550e6,
    1550e6)
}

