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

## Notes

This package provides a native R implementation of OpenAI's Whisper
speech-to-text model. Model weights are downloaded from HuggingFace
on first use (ranging from 145MB for tiny to 3GB for large-v3).
