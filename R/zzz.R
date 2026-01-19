#' @importFrom utils read.csv download.file
#' @importFrom stats setNames
NULL

#' Package Initialization
#'
#' @param libname Library name
#' @param pkgname Package name
.onLoad <- function(
  libname,
  pkgname
) {
  # Set default cache directory for model weights
  cache_dir <- Sys.getenv("WHISPER_CACHE_DIR", "")
  if (cache_dir == "") {
    cache_dir <- file.path(Sys.getenv("HOME"), ".cache", "whisper")
  }

  if (!dir.exists(cache_dir)) {
    dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  }

  options(whisper.cache_dir = cache_dir)
}

