#' Model Download Utilities
#'
#' Download Whisper models from HuggingFace using hfhub.

# Approximate model sizes in MB
.model_sizes <- c(tiny = 151, base = 290, small = 967, medium = 3055,
                  `large-v3` = 6174, `large-v3-turbo` = 1618)

# The revision every hub call in this package resolves against.
#
# NULL means hfhub's default, "main" -- a BRANCH, which it resolves through
# `refs/main` in the cache and, failing that, over the network. Passing an exact
# 40-hex commit takes hfhub's fast path instead: it goes straight to
# `snapshots/<revision>/<file>` and never consults `refs/`.
#
# That difference is what lets whisper run against a cache holding ONLY the
# snapshot -- a read-only bind mount of one revision, with no `refs/` and no
# network. It is also the stronger guarantee generally: a branch moves, so a
# deployment pinned to yesterday's weights would silently start resolving
# today's.
#
# A branch name is REFUSED rather than passed through. Accepting one would hand
# hfhub a value it resolves the slow way, which is the behaviour this argument
# exists to avoid, and the caller would have no way to tell.
.whisper_rev <- function(revision) {
    if (is.null(revision)) {
        return(list())
    }
    if (!is.character(revision) || length(revision) != 1L || is.na(revision) ||
        !grepl("^[0-9a-f]{40}$", revision)) {
        stop("revision must be a single 40-character hex commit, not a branch ",
             "name: a branch resolves through refs/ and defeats the point of ",
             "pinning one", call. = FALSE)
    }
    list(revision = revision)
}

#' Get Model Cache Path
#'
#' @param model Model name
#' @param revision Optional exact 40-hex commit to resolve against.
#' @return Path to model directory in hfhub cache
get_model_path <- function(model, revision = NULL) {
    config <- whisper_config(model)
    repo <- config$hf_repo

    # Use hfhub's cache directory structure
    do.call(hfhub::hub_snapshot,
            c(list(repo, local_files_only = TRUE, allow_patterns = NULL),
              .whisper_rev(revision)))
}

#' Check if Model is Downloaded
#'
#' @param model Model name
#' @param revision Optional exact 40-hex commit to resolve against. With one,
#'   this answers about that snapshot and needs neither a \code{refs/} entry
#'   nor the network.
#' @return TRUE if model weights exist locally
#' @export
#' @examples
#' model_exists("tiny")
#' model_exists("large-v3")
model_exists <- function(model, revision = NULL) {
    config <- whisper_config(model)
    repo <- config$hf_repo

    # OUTSIDE the tryCatch, and that placement is the whole point. Everything
    # below turns an error into FALSE, which is right for "not cached" and
    # wrong for "you passed a branch name": the caller would read a refusal as
    # an absent model and go download it.
    rev <- .whisper_rev(revision)

    tryCatch({
        # Check if safetensors file is cached
        path <- do.call(hfhub::hub_download,
                        c(list(repo, "model.safetensors", local_files_only = TRUE), rev))
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
#' @param model Model name: "tiny", "base", "small", "medium", "large-v3",
#'   "large-v3-turbo"
#' @param force Re-download even if exists
#' @param revision Optional exact 40-hex commit. Every file is fetched at that
#'   one revision, so the resulting cache entry is a single self-contained
#'   snapshot directory.
#' @return Path to model directory (invisibly)
#' @export
#' @examples
#' \donttest{
#' if (interactive()) {
#'   # Download tiny model (smallest, ~150MB)
#'   download_whisper_model("tiny")
#'
#'   # Download larger model for better accuracy
#'   download_whisper_model("small")
#' }
#' }
download_whisper_model <- function(model = "tiny", force = FALSE,
                                   revision = NULL) {
    config <- whisper_config(model)
    repo <- config$hf_repo

    # Check if already downloaded

    if (!force && model_exists(model, revision = revision)) {
        message("Model '", model, "' is already downloaded.")
        return(invisible(get_model_path(model, revision = revision)))
    }

    # Get model size for user info

    size_mb <- .model_sizes[[model]]
    if (!is.null(size_mb)) {
        size_str <- paste0("~", size_mb, " MB")
    } else {
        size_str <- "unknown size"
    }

    # Ask for consent (required for CRAN compliance)
    # Skip prompt if whisper.consent option is set (e.g., from Shiny modal)
    if (isTRUE(getOption("whisper.consent"))) {
        # Consent already given programmatically
    } else if (interactive()) {
        ans <- utils::askYesNo(
                               paste0("Download '", model, "' model (", size_str,
                                      ") from HuggingFace?"),
                               default = TRUE
        )
        if (!isTRUE(ans)) {
            stop("Download cancelled.", call. = FALSE)
        }
    } else {
        stop(
             "Cannot download model in non-interactive mode without consent. ",
             "Run download_whisper_model('", model, "') interactively first, ",
             "or set options(whisper.consent = TRUE) to allow downloads.",
             call. = FALSE
        )
    }

    message("Downloading ", model, " model from HuggingFace (", repo, ")...")

    # Files to download
    files <- c("model.safetensors", "config.json", "vocab.json", "merges.txt")

    # Download all files
    weights_path <- NULL
    for (f in files) {
        message("  ", f, "...")
        tryCatch({
            path <- do.call(hfhub::hub_download,
                            c(list(repo, f, force_download = force),
                              .whisper_rev(revision)))
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
#' @param revision Optional exact 40-hex commit to resolve against.
#' @return Path to safetensors file
get_weights_path <- function(model, revision = NULL) {
    config <- whisper_config(model)
    repo <- config$hf_repo

    # Outside the tryCatch below, which reports every error as a missing
    # download -- a branch name is a caller mistake, not an absent file.
    rev <- .whisper_rev(revision)

    path <- tryCatch({
        do.call(hfhub::hub_download,
                c(list(repo, "model.safetensors", local_files_only = TRUE), rev))
    }, error = function(e) {
        stop("Model weights not found. Run download_whisper_model('", model,
             "') first.")
    })

    # A requested revision is a claim about WHICH bytes load. Check the resolved
    # path against it rather than trusting the argument: the answer comes from
    # where the file actually is, so a cache laid out some other way is refused
    # here instead of loading the wrong weights under the right name.
    if (!is.null(revision)) {
        got <- .snapshot_revision(path)
        if (is.na(got) || !identical(got, revision)) {
            stop("asked for revision ", revision, " but the weights resolved to ",
                 path, call. = FALSE)
        }
    }
    path
}

#' List Available Models
#'
#' @return Character vector of model names
#' @export
#' @examples
#' list_whisper_models()
list_whisper_models <- function() {
    c("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")
}

#' List Downloaded Models
#'
#' @return Character vector of downloaded model names
#' @export
#' @examples
#' list_downloaded_models()
list_downloaded_models <- function() {
    models <- list_whisper_models()
    downloaded <- sapply(models, model_exists)
    models[downloaded]
}
