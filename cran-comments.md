## R CMD check results

0 errors | 0 warnings | 0 notes

On Windows R-devel with a working torch installation the check is clean. On
our Linux development machine torch's Lantern shared library currently fails
to load, which produces two NOTEs of the form "torch failed to start"; those
are an artifact of that machine, not of the package, and do not appear where
torch loads correctly.

## Test environments

* Windows 10, R-devel (2026-07) -- Status: OK
* GitHub Actions (r-ci): ubuntu-latest, macos-latest -- passing
* local Ubuntu 24.04, R 4.6.1 (torch installation broken; see above)

The new R >= 4.5.0 requirement was verified directly against local R 4.5.1
and R 4.4.3: `tools::sha256sum()` is present in the former and absent in the
latter, so the floor is the correct one.

## Changes since last CRAN release (0.4.0)

- `transcribe()` results now carry the shape subtitle tooling expects, so
  they feed `subtitles::whisper_to_srt()` and `whisper_to_ass()` directly. A
  result with segments gains a `data` frame of `from`/`to` timestamp strings
  and `text`, and class `c("whisper_result", "whisper_transcription")`. The
  change is additive: `text`, `segments` and `words` are unchanged, and
  results without segments are returned exactly as before.

- New in-process model residency: a model's weights can be held as
  page-locked (pinned) CPU tensors while its GPU representation is created
  and destroyed on demand, so switching models on a small GPU is a DMA copy
  rather than a reload from disk. New exported functions `resident_load()`,
  `resident_activate()`, `resident_deactivate()`, `resident_transcribe()`,
  `resident_status()`, `resident_unload()`, and a print method for the
  handle. All of it is CUDA-only and inert without a GPU; the examples are
  wrapped so they never execute during checks.

- `resident_deactivate(release = )` selects whether the CUDA caching
  allocator returns its blocks to the driver (the default, unchanged
  semantics) or retains them for the next model in the same process.
  Measured on a 6 GB card, retaining the pool makes reactivation about ten
  times faster.

- The package now requires R >= 4.5.0, for `tools::sha256sum()`, which gives
  each loaded model a content-addressed identity.

## Notes

This package provides a native R implementation of OpenAI's Whisper
speech-to-text model. Model weights are downloaded from HuggingFace on
first use (145MB for tiny to 3GB for large-v3). GPU/CUDA-specific code
paths (JIT decode, GC tuning, the fp16 fallback, residency, `serve()`) are
gated and do not run during checks on machines without CUDA.

## Reverse dependencies

None on CRAN.
