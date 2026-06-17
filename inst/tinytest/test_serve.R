# The multipart/form-data parser must be binary-safe (audio uploads) and
# extract both the file part and string fields the way curl sends them.
# Pure R, no model/CUDA, so it runs during R CMD check.

boundary <- "----WhisperBoundary7MA4YWxkTrZu0gW"

# Binary file payload including bytes that must survive a round trip
# (NUL, CR, LF, 0xFF) - proves we never coerce the body to a string.
file_bytes <- as.raw(c(0x00, 0x52, 0x49, 0x46, 0x46, 0xFF, 0x0D, 0x0A, 0x7A))

crlf <- "\r\n"
head <- charToRaw(paste0(
  "--", boundary, crlf,
  'Content-Disposition: form-data; name="file"; filename="clip.wav"', crlf,
  "Content-Type: audio/wav", crlf, crlf))
mid <- charToRaw(paste0(crlf,
  "--", boundary, crlf,
  'Content-Disposition: form-data; name="response_format"', crlf, crlf,
  "verbose_json", crlf,
  "--", boundary, crlf,
  'Content-Disposition: form-data; name="language"', crlf, crlf,
  "en", crlf,
  "--", boundary, "--", crlf))
body <- c(head, file_bytes, mid)

parts <- whisper:::.serve_parse_multipart(body, boundary)

expect_true(!is.null(parts[["file"]]), info = "file part found")
expect_identical(parts[["file"]]$value, file_bytes,
  info = "binary file bytes survive round trip")
expect_identical(parts[["file"]]$filename, "clip.wav",
  info = "filename parsed")
expect_identical(rawToChar(parts[["response_format"]]$value), "verbose_json",
  info = "string field parsed")
expect_identical(rawToChar(parts[["language"]]$value), "en",
  info = "second string field parsed")

# A body with no parts / bad boundary returns an empty list, not an error
expect_identical(whisper:::.serve_parse_multipart(raw(0), boundary), list())
expect_identical(whisper:::.serve_parse_multipart(body, "nomatch"), list())

# Boundary extraction from a Content-Type header (quoted and unquoted)
extract_boundary <- function(ct) {
  trimws(gsub('"', "", sub(".*boundary=([^;]+).*", "\\1", ct)))
}
ct1 <- paste0("multipart/form-data; boundary=", boundary)
expect_identical(extract_boundary(ct1), boundary)
ct2 <- paste0('multipart/form-data; boundary="', boundary, '"')
expect_identical(extract_boundary(ct2), boundary)

# Segment serialization: empty -> empty list, rows -> records
expect_identical(whisper:::.serve_segments(NULL), list())
empty_df <- data.frame(start = numeric(0), end = numeric(0),
  text = character(0))
expect_identical(whisper:::.serve_segments(empty_df), list())
df <- data.frame(start = c(0, 1.5), end = c(1.5, 3),
  text = c("a", "b"), stringsAsFactors = FALSE)
segs <- whisper:::.serve_segments(df)
expect_equal(length(segs), 2L)
expect_equal(segs[[2]]$start, 1.5)
expect_identical(segs[[1]]$text, "a")
