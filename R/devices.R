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
#' Returns float16 on CUDA, float32 on CPU.
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
  # Use float16 on CUDA, float32 on CPU

  if (device$type == "cuda") {
    torch::torch_float16()
  } else {
    torch::torch_float()
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

