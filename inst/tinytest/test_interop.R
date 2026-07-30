# Tests for the subtitle-tool interop shape (R/interop.R).
# Pure data reshaping, so none of this needs torch or a model.

# ---- format_hms ----

expect_equal(whisper:::format_hms(0), "00:00:00.000")
expect_equal(whisper:::format_hms(7.4), "00:00:07.400")
expect_equal(whisper:::format_hms(61.25), "00:01:01.250")
expect_equal(whisper:::format_hms(3661.5), "01:01:01.500")
expect_equal(whisper:::format_hms(c(0, 7.4)), c("00:00:00.000", "00:00:07.400"))

# Rounds to whole milliseconds before splitting, so a value just under a
# boundary carries instead of printing a 60th second/minute.
expect_equal(whisper:::format_hms(59.9996), "00:01:00.000")
expect_equal(whisper:::format_hms(3599.9999), "01:00:00.000")

# Negative and non-finite input clamps to zero rather than erroring.
# sprintf()'s %d rejects non-finite doubles, so Inf has to be caught by the
# guard and not just NA -- is.na() alone lets Inf through to an error.
expect_equal(whisper:::format_hms(-1), "00:00:00.000")
expect_equal(whisper:::format_hms(NA_real_), "00:00:00.000")
expect_equal(whisper:::format_hms(NaN), "00:00:00.000")
expect_equal(whisper:::format_hms(Inf), "00:00:00.000")
expect_equal(whisper:::format_hms(-Inf), "00:00:00.000")
expect_equal(whisper:::format_hms(c(1.5, Inf, NA_real_, -2)),
             c("00:00:01.500", "00:00:00.000", "00:00:00.000",
               "00:00:00.000"))

# Every output is a well-formed HH:MM:SS.mmm string.
expect_true(all(grepl("^\\d{2}:\\d{2}:\\d{2}\\.\\d{3}$",
                      whisper:::format_hms(c(0, 1.0005, 59.9996, 3661.5)))))

# ---- attach_subtitle_shape ----

res <- list(text = "one two", language = "en",
            segments = data.frame(start = c(0, 2.5), end = c(2.5, 5),
                                  text = c("one", "two"),
                                  stringsAsFactors = FALSE),
            model = "tiny", backend = "whisper", duration = 5)

out <- whisper:::attach_subtitle_shape(res)

# The class leads with our own, so audio.whisper's methods on
# "whisper_transcription" stay a fallback rather than capturing our objects.
expect_equal(class(out), c("whisper_result", "whisper_transcription"))
expect_true(inherits(out, "whisper_transcription"))

# $data is what the subtitle writers read.
expect_true(is.data.frame(out$data))
expect_equal(names(out$data), c("from", "to", "text"))
expect_equal(out$data$from, c("00:00:00.000", "00:00:02.500"))
expect_equal(out$data$to, c("00:00:02.500", "00:00:05.000"))
expect_equal(out$data$text, c("one", "two"))

# Purely additive: nothing that was there before moved or changed.
expect_equal(out$text, res$text)
expect_equal(out$language, res$language)
expect_equal(out$segments, res$segments)
expect_equal(out$model, res$model)
expect_equal(out$duration, res$duration)

# Word timings ride through untouched for karaoke.
res_w <- res
res_w$words <- data.frame(word = c("one", "two"), start = c(0, 2.5),
                          end = c(0.5, 3), stringsAsFactors = FALSE)
expect_equal(whisper:::attach_subtitle_shape(res_w)$words, res_w$words)

# ---- no segments: returned unchanged and unclassed ----
# timestamps = FALSE produces this shape. Claiming the class without $data
# would fail inside subtitles::whisper_to_srt().

bare <- list(text = "one two", language = "en", model = "tiny")
expect_identical(whisper:::attach_subtitle_shape(bare), bare)
expect_false(inherits(whisper:::attach_subtitle_shape(bare),
                      "whisper_transcription"))

empty <- list(text = "", language = "en",
              segments = data.frame(start = numeric(0), end = numeric(0),
                                    text = character(0)))
expect_identical(whisper:::attach_subtitle_shape(empty), empty)
expect_false(inherits(whisper:::attach_subtitle_shape(empty),
                      "whisper_transcription"))

# ---- print method ----

expect_stdout(print(out), "whisper transcription")
expect_equal(print(out), out)

# ---- the contract subtitles:: checks ----
# subtitles::whisper_to_srt()/whisper_to_ass() do exactly two things with the
# object they are handed: stopifnot(inherits(x, "whisper_transcription")), then
# read x$data's from/to/text. Asserting that contract here keeps the test
# honest without making an STT package depend on a subtitle package -- the
# coupling is a data shape, not a package dependency.

expect_true(inherits(out, "whisper_transcription"))
expect_true(all(c("from", "to", "text") %in% names(out$data)))
expect_true(is.character(out$data$from))
expect_true(is.character(out$data$to))
expect_true(is.character(out$data$text))
expect_equal(nrow(out$data), nrow(out$segments))
expect_true(all(grepl("^\\d{2}:\\d{2}:\\d{2}\\.\\d{3}$",
                      c(out$data$from, out$data$to))))
