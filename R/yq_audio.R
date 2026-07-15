# anvl/yunque log-mel frontend, a torch-free port of audio_to_mel().
# The yq_ prefix marks the anvl/XLA implementation (vs the torch one).
# Numeric core (.yq_log_mel) is separated from I/O so it can be verified
# in an anvl-only session against a torch fixture.

# Padded-audio vector + mel filterbank -> log-mel AnvlArray [1, n_mels, 3000].
# Mirrors audio_to_mel: STFT -> power -> mel -> clamp/log10 -> normalize.
.yq_log_mel <- function(audio, mel_fb) {
  win <- yunque::hann_window(WHISPER_N_FFT)
  sig <- anvl::nv_array(matrix(as.numeric(audio), nrow = 1L), dtype = "f32")
  sp <- yunque::stft(sig, WHISPER_N_FFT, WHISPER_HOP_LENGTH,
    window = win, center = TRUE)
  nfr <- anvl::shape(sp$real)[3L] - 1L # drop the trailing frame (Whisper conv.)
  re <- yunque::slice_lastdim(sp$real, 1L, nfr)
  im <- yunque::slice_lastdim(sp$imag, 1L, nfr)
  power <- re * re + im * im # |stft|^2, [1, n_freqs, nfr]
  fb <- anvl::nv_unsqueeze(anvl::nv_array(mel_fb, dtype = "f32"), 1L)
  mel <- anvl::nv_matmul(fb, power) # [1, n_mels, nfr]
  mel <- anvl::nv_max(mel, 1e-10) # clamp min
  log_mel <- anvl::nv_log10(mel)
  mx <- anvl::nv_reduce_max(log_mel, dims = seq_len(anvl::ndims(log_mel)),
    drop = TRUE)
  log_mel <- anvl::nv_max(log_mel, mx - 8)
  (log_mel + 4) / 4
}

#' Audio to log-mel spectrogram (anvl/yunque)
#'
#' Torch-free port of \code{\link{audio_to_mel}} on the anvl/XLA stack: the
#' STFT is \code{yunque::stft} (an FFT-free windowed DFT), and the mel
#' projection, log, and normalization run as anvl ops.
#'
#' @param file Path to an audio file, or a numeric vector of samples.
#' @param n_mels Number of mel bins (80, or 128 for large-v3).
#'
#' @return AnvlArray \code{[1, n_mels, 3000]}.
#'
#' @export
yq_audio_to_mel <- function(file, n_mels = 80L) {
  audio <- if (is.character(file)) load_audio(file) else as.numeric(file)
  audio <- pad_or_trim(audio)
  mel_fb <- load_mel_filterbank(n_mels = n_mels)
  .yq_log_mel(audio, mel_fb)
}
