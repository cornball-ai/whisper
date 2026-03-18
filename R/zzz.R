#' @importFrom utils read.csv download.file
#' @importFrom stats setNames
NULL

.onAttach <- function(libname, pkgname) {
    if (!torch::torch_is_installed()) {
        packageStartupMessage(
            "torch backend (Lantern) is not installed.\n",
            "Run torch::install_torch() to complete setup."
        )
    }
}

