# serve.R
# Minimal HTTP server exposing whisper over an OpenAI-compatible
# /v1/audio/transcriptions endpoint. Built on base R sockets
# (serverSocket/socketAccept) so it adds no dependencies and runs as a
# single persistent process: the model loads once and stays resident on
# the GPU (no fork, so the CUDA context is never invalidated). Requests
# are served one at a time, the natural fit for a single-GPU box.
#
# The request is multipart/form-data (OpenAI's audio API uploads the file
# as a part), unlike the chatterbox TTS server's JSON body; the multipart
# parser below is the only substantive difference. The contract matches
# what stt.api's API backend sends: fields file/model/language/prompt/
# response_format, and a {text, language, segments:[{start,end,text}]}
# response (plain text when response_format = "text").

#' Serve whisper over HTTP
#'
#' Starts a blocking HTTP server that loads a whisper model once and
#' answers OpenAI-compatible speech-to-text requests. Intended as a
#' drop-in for the OpenAI transcription API or a Whisper container: point
#' an HTTP client (e.g. \pkg{stt.api} via \code{set_stt_base()}) at
#' \code{http://<host>:<port>} and it serves the same endpoint.
#'
#' Endpoints:
#' \itemize{
#'   \item \code{GET /health} - liveness probe, returns
#'     \code{{"status":"ok","model":...}}.
#'   \item \code{POST /v1/audio/transcriptions} - multipart/form-data with
#'     fields \code{file} (required audio upload), \code{language},
#'     \code{response_format} (\code{json} (default), \code{text}, or
#'     \code{verbose_json}), and \code{temperature}. Returns the
#'     transcription. \code{verbose_json} adds \code{segments} with
#'     start/end times.
#'   \item \code{POST /v1/audio/translations} - same, but translates to
#'     English (Whisper's translate task).
#' }
#'
#' The server is single-threaded and runs until interrupted. Run it under
#' a process supervisor (systemd, a container CMD, tmux) for persistence;
#' an example systemd unit ships with the package:
#' \code{system.file("whisper.service", package = "whisper")}. It is
#' designed to sit alongside a chatterbox TTS server as a second always-on
#' process on the same GPU (each has its own CUDA context).
#' On CUDA it tunes torch's allocator GC (see \code{\link{whisper_tune_gc}})
#' before loading and uses the TorchScript greedy decode step.
#'
#' @param port Integer. TCP port to listen on. Default 4123.
#' @param model Model name to load and keep resident (the request's
#'   \code{model} field is ignored). Default "large-v3".
#' @param device Character. Torch device ("cuda", "cpu", "mps").
#' @param dtype Compute dtype ("auto", "float16", "float32").
#' @param timeout Integer. Per-connection I/O timeout in seconds (guards
#'   against stalled clients). Default 300.
#' @param max_body Integer. Maximum request body size in bytes. Default
#'   100 MB (audio uploads are larger than JSON bodies).
#' @param warmup Logical. Transcribe a short bundled clip at startup to
#'   compile the decode step and prime the allocator, so the first client
#'   request isn't slow. Default TRUE.
#' @return Does not return normally; runs until interrupted.
#' @export
serve <- function(port = 4123L, model = "large-v3", device = "cuda",
                  dtype = "auto", timeout = 300L,
                  max_body = 100L * 1024L ^ 2, warmup = TRUE) {
  # Tune the CUDA allocator GC before the first CUDA op (no-op off CUDA).
  whisper_tune_gc(model, device = device, dtype = dtype)

  message("Loading whisper model '", model, "' on ", device, " ...")
  pipe <- whisper_pipeline(model, device = device, dtype = dtype,
    download = TRUE, verbose = FALSE)
  message("Model loaded.")

  if (isTRUE(warmup)) {
    sample <- system.file("audio", "jfk.mp3", package = "whisper")
    if (nzchar(sample)) {
      message("Warming up ...")
      tryCatch(
        invisible(pipe$transcribe(sample, language = "en", timestamps = TRUE,
          verbose = FALSE)),
        error = function(e) message("warmup skipped: ", conditionMessage(e)))
      message("Warmup done.")
    }
  }

  srv <- serverSocket(port)
  on.exit(close(srv), add = TRUE)
  message("whisper::serve listening on port ", port, " (interrupt to stop)")

  repeat {
    con <- tryCatch(
      socketAccept(srv, blocking = TRUE, open = "r+b", timeout = timeout),
      error = function(e) {
        message("accept error: ", conditionMessage(e))
        Sys.sleep(0.5)  # avoid busy-spin on a bad server socket
        NULL
      })
    if (is.null(con)) {
      next
    }
    tryCatch({
      req <- .serve_read_request(con, max_body)
      if (!is.null(req)) {
        resp <- tryCatch(
          .serve_route(req, pipe, model),
          error = function(e) .serve_err(500L, conditionMessage(e)))
        .serve_send(con, resp$status, resp$content_type, resp$body)
      }
    },
      error = function(e) message("request error: ", conditionMessage(e)),
      finally = {
        try(close(con), silent = TRUE)
        gc()  # bound dead tensor handles per request; keep the warm pool
      })
  }
}

# Null/empty/blank coalescing
.serve_or <- function(a, b) {
  if (is.null(a) || length(a) == 0L ||
      (is.character(a) && !nzchar(a))) b else a
}

# Read and parse one HTTP request from a connection. Returns a list with
# method/path/headers/body, or NULL on a closed/incomplete/oversized header.
.serve_read_request <- function(con, max_body) {
  term <- as.raw(c(13L, 10L, 13L, 10L))  # CRLFCRLF
  buf <- raw(0)
  max_header <- 65536L
  repeat {
    b <- readBin(con, "raw", n = 1L)
    if (length(b) == 0L) {
      return(NULL)  # closed or timed out mid-header
    }
    buf <- c(buf, b)
    n <- length(buf)
    if (n >= 4L && identical(buf[(n - 3L):n], term)) {
      break
    }
    if (n > max_header) {
      return(NULL)
    }
  }

  lines <- strsplit(rawToChar(buf), "\r\n", fixed = TRUE)[[1]]
  req_line <- strsplit(lines[1], " ", fixed = TRUE)[[1]]
  if (length(req_line) < 2L) {
    return(NULL)
  }
  method <- req_line[1]
  path <- req_line[2]

  hdr <- list()
  if (length(lines) > 1L) {
    for (ln in lines[-1L]) {
      if (!nzchar(ln)) {
        next
      }
      pos <- regexpr(":", ln, fixed = TRUE)
      if (pos < 1L) {
        next
      }
      key <- tolower(trimws(substr(ln, 1L, pos - 1L)))
      hdr[[key]] <- trimws(substr(ln, pos + 1L, nchar(ln)))
    }
  }

  cl <- hdr[["content-length"]]
  if (is.null(cl)) {
    clen <- 0L
  } else {
    clen <- suppressWarnings(as.integer(cl))
  }
  if (length(clen) != 1L || is.na(clen) || clen < 0L) {
    clen <- 0L
  }
  if (clen > max_body) {
    return(list(method = method, path = path, headers = hdr,
      body = raw(0), too_large = TRUE))
  }

  if (clen > 0L) {
    body <- .serve_read_n(con, clen)
  } else {
    body <- raw(0)
  }
  list(method = method, path = path, headers = hdr, body = body)
}

# Read exactly n bytes (or until the stream ends).
.serve_read_n <- function(con, n) {
  out <- raw(0)
  while (length(out) < n) {
    chunk <- readBin(con, "raw", n = n - length(out))
    if (length(chunk) == 0L) {
      break
    }
    out <- c(out, chunk)
  }
  out
}

# Write an HTTP/1.1 response and close (Connection: close).
.serve_send <- function(con, status, content_type, body) {
  if (is.character(body)) {
    body <- charToRaw(enc2utf8(body))
  }
  reason <- switch(as.character(status), "200" = "OK",
    "400" = "Bad Request", "404" = "Not Found",
    "405" = "Method Not Allowed",
    "413" = "Payload Too Large",
    "500" = "Internal Server Error", "Unknown")
  head <- paste0(
    sprintf("HTTP/1.1 %d %s\r\n", status, reason),
    sprintf("Content-Type: %s\r\n", content_type),
    sprintf("Content-Length: %d\r\n", length(body)),
    "Connection: close\r\n\r\n")
  writeBin(c(charToRaw(head), body), con)
  flush(con)
}

# Dispatch a parsed request to a handler.
.serve_route <- function(req, pipe, model) {
  if (isTRUE(req$too_large)) {
    return(.serve_err(413L, "request body too large"))
  }
  path <- sub("\\?.*$", "", req$path)

  if (identical(req$method, "GET") && path == "/health") {
    return(.serve_json(list(status = "ok", model = model)))
  }
  if (identical(req$method, "POST") && path == "/v1/audio/transcriptions") {
    return(.serve_transcribe(req, pipe, task = "transcribe"))
  }
  if (identical(req$method, "POST") && path == "/v1/audio/translations") {
    return(.serve_transcribe(req, pipe, task = "translate"))
  }
  .serve_err(404L, "not found")
}

# Handle a transcription/translation request.
.serve_transcribe <- function(req, pipe, task = "transcribe") {
  ct <- .serve_or(req$headers[["content-type"]], "")
  if (!grepl("multipart/form-data", ct, ignore.case = TRUE)) {
    return(.serve_err(400L, "expected multipart/form-data"))
  }
  if (!grepl("boundary=", ct, fixed = TRUE)) {
    return(.serve_err(400L, "missing multipart boundary"))
  }
  # Base R regex (TRE) has no pattern backreferences, so strip any quotes
  # in a second step rather than matching them with \\1.
  boundary <- trimws(gsub('"', "", sub(".*boundary=([^;]+).*", "\\1", ct)))
  if (!nzchar(boundary)) {
    return(.serve_err(400L, "missing multipart boundary"))
  }

  parts <- .serve_parse_multipart(req$body, boundary)
  filepart <- parts[["file"]]
  if (is.null(filepart) || length(filepart$value) == 0L) {
    return(.serve_err(400L, "'file' field is required"))
  }

  # Persist the upload to a temp file with the original extension so the
  # av/ffmpeg decoder picks the right container.
  ext <- tolower(tools::file_ext(.serve_or(filepart$filename, "")))
  if (!nzchar(ext)) {
    ext <- "wav"
  }
  tmp <- tempfile(fileext = paste0(".", ext))
  on.exit(unlink(tmp), add = TRUE)
  writeBin(filepart$value, tmp)

  fld <- function(k) {
    if (!is.null(parts[[k]])) rawToChar(parts[[k]]$value) else NULL
  }
  language <- .serve_or(fld("language"), NULL)
  rf <- tolower(.serve_or(fld("response_format"), "json"))
  want_ts <- rf == "verbose_json"

  args <- list(file = tmp, task = task, language = language,
    timestamps = want_ts, verbose = FALSE)
  temp <- suppressWarnings(as.numeric(.serve_or(fld("temperature"), "")))
  if (length(temp) == 1L && !is.na(temp)) {
    args$temperatures <- temp
  }
  res <- do.call(pipe$transcribe, args)

  if (rf == "text") {
    return(list(status = 200L,
      content_type = "text/plain; charset=utf-8",
      body = paste0(res$text, "\n")))
  }

  obj <- list(text = res$text, language = res$language)
  if (want_ts) {
    obj$task <- task
    obj$duration <- res$duration
    obj$segments <- .serve_segments(res$segments)
  }
  .serve_json(obj)
}

# Parse a multipart/form-data body into a named list of parts. Each part
# is list(value = raw, filename = character|NA). Binary-safe: the body is
# split on the boundary with grepRaw (C-level), never coerced to a string.
.serve_parse_multipart <- function(body, boundary) {
  if (length(body) == 0L) {
    return(list())
  }
  delim <- charToRaw(paste0("--", boundary))
  pos <- grepRaw(delim, body, all = TRUE)
  if (length(pos) < 2L) {
    return(list())
  }
  dlen <- length(delim)
  hdr_term <- as.raw(c(13L, 10L, 13L, 10L))
  crlf <- as.raw(c(13L, 10L))
  parts <- list()

  for (i in seq_len(length(pos) - 1L)) {
    seg <- body[(pos[i] + dlen):(pos[i + 1L] - 1L)]
    if (length(seg) < 2L) {
      next
    }
    # The closing delimiter is "--boundary--": its segment starts with "--".
    if (seg[1] == as.raw(0x2d) && seg[2] == as.raw(0x2d)) {
      next
    }
    # Strip the leading CRLF after the boundary and the trailing CRLF.
    if (seg[1] == crlf[1] && seg[2] == crlf[2]) {
      seg <- seg[-(1:2)]
    }
    n <- length(seg)
    if (n >= 2L && seg[n - 1L] == crlf[1] && seg[n] == crlf[2]) {
      seg <- seg[1:(n - 2L)]
    }
    hp <- grepRaw(hdr_term, seg, all = FALSE)
    if (length(hp) == 0L) {
      next
    }
    hdr_txt <- rawToChar(seg[1:(hp - 1L)])
    content <- if ((hp + 4L) <= length(seg)) {
      seg[(hp + 4L):length(seg)]
    } else {
      raw(0)
    }

    cd <- grep("content-disposition", strsplit(hdr_txt, "\r\n", fixed = TRUE)[[1]],
      ignore.case = TRUE, value = TRUE)[1]
    if (is.na(cd)) {
      next
    }
    # Require a delimiter before "name=" so the greedy match does not grab
    # the value of "filename=" instead.
    name <- if (grepl('[ ;]name="', cd)) {
      sub('.*[ ;]name="([^"]*)".*', "\\1", cd)
    } else {
      NA
    }
    filename <- if (grepl('filename="', cd)) {
      sub('.*filename="([^"]*)".*', "\\1", cd)
    } else {
      NA
    }
    if (is.na(name)) {
      next
    }
    parts[[name]] <- list(value = content, filename = filename)
  }
  parts
}

# Convert a segments data.frame (start, end, text) to a list of records so
# jsonlite renders it as an array of objects; empty -> [].
.serve_segments <- function(segments) {
  if (is.null(segments) || nrow(segments) == 0L) {
    return(list())
  }
  lapply(seq_len(nrow(segments)), function(i) {
    list(start = segments$start[i], end = segments$end[i],
      text = segments$text[i])
  })
}

# JSON 200 response.
.serve_json <- function(obj) {
  list(status = 200L, content_type = "application/json",
    body = jsonlite::toJSON(obj, auto_unbox = TRUE, na = "null"))
}

# JSON error response in the OpenAI {"error":{"message":...}} shape.
.serve_err <- function(status, msg) {
  list(status = status, content_type = "application/json",
    body = jsonlite::toJSON(list(error = list(message = msg)),
      auto_unbox = TRUE))
}
