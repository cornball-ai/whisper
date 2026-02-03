## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

## Test environments

* local Ubuntu 24.04, R 4.4.x
* GitHub Actions (r-ci): ubuntu-latest, macos-latest

## Package dependencies

This package requires:
- torch: Neural network operations
- av: Audio file loading (FFmpeg bindings)
- jsonlite: JSON parsing for tokenizer files
- hfhub: HuggingFace model downloads
- safetensors: Model weight loading

## Resubmission changes

- Added cornball.ai as copyright holder (`cph`) in `Authors@R` to match
  the LICENSE file.
- Added OpenAI as copyright holder (`cph`) in `Authors@R`. The bundled mel
  filterbank data (`inst/assets/mel_80.csv`, `mel_128.csv`) is from OpenAI's
  MIT-licensed Whisper repository, and the model architecture is derived from
  their specifications.
- Changed `\dontrun{}` to `\donttest{}` in `download_whisper_model()` examples.

## Notes

This package provides a native R implementation of OpenAI's Whisper
speech-to-text model. Model weights are downloaded from HuggingFace
on first use (ranging from 145MB for tiny to 3GB for large-v3).
