#' Model Download Utilities
#'
#' Download Whisper models from HuggingFace using hfhub.

# Approximate model sizes in MB
.model_sizes <- c(

  tiny = 151,
  base = 290,
  small = 967,
  medium = 3055,
  `large-v3` = 6174
)

#' Get Model Cache Path
#'
#' @param model Model name
#' @return Path to model directory in hfhub cache
get_model_path <- function(model) {
  config <- whisper_config(model)
  repo <- config$hf_repo

  # Use hfhub's cache directory structure
  hfhub::hub_snapshot(repo, local_files_only = TRUE, allow_patterns = NULL)
}

#' Check if Model is Downloaded
#'
#' @param model Model name
#' @return TRUE if model weights exist locally
#' @export
model_exists <- function(model) {
  config <- whisper_config(model)
  repo <- config$hf_repo

  tryCatch({
      # Check if safetensors file is cached
      path <- hfhub::hub_download(repo, "model.safetensors", local_files_only = TRUE)
      file.exists(path)
    }, error = function(e) {
      FALSE
    })
}

#' Download Model from HuggingFace
#'
#' Download Whisper model weights and tokenizer files from HuggingFace.
#' In interactive sessions, asks for user consent before downloading.
#'
#' @param model Model name: "tiny", "base", "small", "medium", "large-v3"
#' @param force Re-download even if exists
#' @return Path to model directory (invisibly)
#' @export
download_whisper_model <- function(
  model = "tiny",
  force = FALSE
) {
  config <- whisper_config(model)
  repo <- config$hf_repo

  # Check if already downloaded

  if (!force && model_exists(model)) {
    message("Model '", model, "' is already downloaded.")
    return(invisible(get_model_path(model)))
  }

  # Get model size for user info

  size_mb <- .model_sizes[[model]]
  size_str <- if (!is.null(size_mb)) paste0("~", size_mb, " MB") else "unknown size"

  # Ask for consent in interactive mode

  if (interactive()) {
    ans <- utils::askYesNo(
      paste0("Download '", model, "' model (", size_str, ") from HuggingFace?"),
      default = TRUE
    )
    if (!isTRUE(ans)) {
      stop("Download cancelled.", call. = FALSE)
    }
  }

  message("Downloading ", model, " model from HuggingFace (", repo, ")...")

  # Files to download
  files <- c(
    "model.safetensors",
    "config.json",
    "vocab.json",
    "merges.txt"
  )

  # Download all files
  weights_path <- NULL
  for (f in files) {
    message("  ", f, "...")
    tryCatch({
        path <- hfhub::hub_download(repo, f, force_download = force)
        if (f == "model.safetensors") weights_path <- path
      }, error = function(e) {
        warning("Failed to download ", f, ": ", e$message)
      })
  }

  if (is.null(weights_path)) {
    stop("Failed to download model weights")
  }

  model_path <- dirname(weights_path)
  message("Model downloaded to: ", model_path)
  invisible(model_path)
}

#' Get Path to Model Weights
#'
#' @param model Model name
#' @return Path to safetensors file
get_weights_path <- function(model) {
  config <- whisper_config(model)
  repo <- config$hf_repo

  tryCatch({
      hfhub::hub_download(repo, "model.safetensors", local_files_only = TRUE)
    }, error = function(e) {
      stop("Model weights not found. Run download_whisper_model('", model, "') first.")
    })
}

#' List Available Models
#'
#' @return Character vector of model names
#' @export
list_whisper_models <- function() {
  c("tiny", "base", "small", "medium", "large-v3")
}

#' List Downloaded Models
#'
#' @return Character vector of downloaded model names
#' @export
list_downloaded_models <- function() {
  models <- list_whisper_models()
  downloaded <- sapply(models, model_exists)
  models[downloaded]
}

