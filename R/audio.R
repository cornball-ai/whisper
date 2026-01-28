#' Audio Preprocessing for Whisper
#'
#' Convert audio files to mel spectrograms for Whisper input.

#' Whisper Audio Constants
WHISPER_SAMPLE_RATE <- 16000L
WHISPER_N_FFT <- 400L
WHISPER_HOP_LENGTH <- 160L
WHISPER_CHUNK_LENGTH <- 30L# seconds
WHISPER_N_SAMPLES <- WHISPER_CHUNK_LENGTH * WHISPER_SAMPLE_RATE# 480000

#' Load and Preprocess Audio
#'
#' Load audio from file, convert to mono, resample to 16kHz.
#'
#' @param file Path to audio file (WAV, MP3, etc.)
#' @return Numeric vector of audio samples normalized to -1 to 1 range
#' @importFrom utils capture.output
#' @export
#' @examples
#' # Load included sample audio
#' audio_file <- system.file("audio", "jfk.mp3", package = "whisper")
#' samples <- load_audio(audio_file)
#' length(samples)
#' range(samples)
load_audio <- function(file) {

  if (!file.exists(file)) {
    stop("Audio file not found: ", file)
  }

  # av::read_audio_bin returns raw PCM samples as 32-bit signed integers
  # channels = 1 for mono, sample_rate = 16000 for Whisper
  # Suppress FFmpeg stderr messages ("Insufficient memory to recode all samples")
  audio <- suppressWarnings(
    invisible(capture.output(
        result <- av::read_audio_bin(file, channels = 1L, sample_rate = WHISPER_SAMPLE_RATE),
        type = "message"
      ))
  )
  audio <- result

  # Normalize 32-bit signed integers to -1 to 1 range
  as.numeric(audio) / 2147483648
}

#' Pad or Trim Audio to Fixed Length
#'
#' @param audio Numeric vector of audio samples
#' @param length Target length in samples (default: 30s at 16kHz)
#' @return Numeric vector of specified length
pad_or_trim <- function(
  audio,
  length = WHISPER_N_SAMPLES
) {
  n <- length(audio)

  if (n > length) {
    # Trim
    audio[seq_len(length)]
  } else if (n < length) {
    # Pad with zeros
    c(audio, rep(0, length - n))
  } else {
    audio
  }
}

#' Load Pre-computed Mel Filterbank
#'
#' Load the official Whisper mel filterbank from bundled CSV file.
#'
#' @param n_mels Number of mel bins (80 or 128)
#' @return Mel filterbank matrix (n_mels x n_freqs)
load_mel_filterbank <- function(n_mels = 80L) {
  csv_file <- system.file("assets", paste0("mel_", n_mels, ".csv"), package = "whisper")
  if (csv_file == "") {
    stop("Mel filterbank file not found for n_mels = ", n_mels)
  }
  as.matrix(read.csv(csv_file, header = FALSE))
}

#' Create Mel Filterbank (Fallback)
#'
#' Create a mel filterbank matrix for converting STFT to mel spectrogram.
#' Used when pre-computed filterbank is not available.
#'
#' @param n_fft FFT size
#' @param n_mels Number of mel bins
#' @param sample_rate Audio sample rate
#' @return Mel filterbank matrix (n_mels x n_freqs)
create_mel_filterbank_fallback <- function(
  n_fft = WHISPER_N_FFT,
  n_mels = 80L,
  sample_rate = WHISPER_SAMPLE_RATE
) {
  # Number of frequency bins
  n_freqs <- n_fft %/% 2L + 1L

  # Frequency range
  f_min <- 0
  f_max <- sample_rate / 2

  # Convert to mel scale
  mel_min <- hz_to_mel(f_min)
  mel_max <- hz_to_mel(f_max)

  # Create mel points evenly spaced in mel scale
  mel_points <- seq(mel_min, mel_max, length.out = n_mels + 2L)
  hz_points <- mel_to_hz(mel_points)

  # Convert to FFT bin indices
  bin_points <- floor((n_fft + 1L) * hz_points / sample_rate)

  # Create filterbank
  filterbank <- matrix(0, nrow = n_mels, ncol = n_freqs)

  for (i in seq_len(n_mels)) {
    left <- bin_points[i]
    center <- bin_points[i + 1L]
    right <- bin_points[i + 2L]

    # Rising slope
    for (j in seq(left, center)) {
      if (j >= 1L && j <= n_freqs && center > left) {
        filterbank[i, j] <- (j - left) / (center - left)
      }
    }

    # Falling slope
    for (j in seq(center, right)) {
      if (j >= 1L && j <= n_freqs && right > center) {
        filterbank[i, j] <- (right - j) / (right - center)
      }
    }
  }

  filterbank
}

#' Convert Hz to Mel Scale
#'
#' @param hz Frequency in Hz
#' @return Frequency in mel scale
hz_to_mel <- function(hz) {
  # Slaney formula (used by Whisper)
  2595 * log10(1 + hz / 700)
}

#' Convert Mel Scale to Hz
#'
#' @param mel Frequency in mel scale
#' @return Frequency in Hz
mel_to_hz <- function(mel) {
  700 * (10 ^ (mel / 2595) - 1)
}

#' Compute STFT Magnitude
#'
#' @param audio Numeric vector of audio samples
#' @param n_fft FFT window size
#' @param hop_length Hop length between frames
#' @return Complex STFT matrix
compute_stft <- function(
  audio,
  n_fft = WHISPER_N_FFT,
  hop_length = WHISPER_HOP_LENGTH
) {
  # Convert to tensor
  audio_tensor <- torch::torch_tensor(audio, dtype = torch::torch_float())

  # Add batch dimension
  audio_tensor <- audio_tensor$unsqueeze(1L)

  # Hann window
  window <- torch::torch_hann_window(n_fft)

  # Compute STFT
  # Returns complex tensor of shape (batch, n_freqs, n_frames)
  stft <- torch::torch_stft(
    audio_tensor,
    n_fft = n_fft,
    hop_length = hop_length,
    win_length = n_fft,
    window = window,
    center = TRUE,
    pad_mode = "reflect",
    normalized = FALSE,
    onesided = TRUE,
    return_complex = TRUE
  )

  # Remove batch dim: (n_freqs, n_frames)
  stft <- stft$squeeze(1L)

  # Remove last frame (Whisper convention)
  stft[, 1:(stft$size(2) - 1L)]
}

#' Convert Audio to Mel Spectrogram
#'
#' Main preprocessing function that converts audio to the mel spectrogram
#' format expected by Whisper.
#'
#' @param file Path to audio file, or numeric vector of audio samples
#' @param n_mels Number of mel bins (80 for most models, 128 for large-v3)
#' @param device torch device for output tensor
#' @param dtype torch dtype for output tensor
#' @return torch tensor of shape (1, n_mels, 3000) for 30s audio
#' @export
#' @examples
#' \donttest{
#' # Convert audio file to mel spectrogram
#' audio_file <- system.file("audio", "jfk.mp3", package = "whisper")
#' mel <- audio_to_mel(audio_file)
#' dim(mel)
#' }
audio_to_mel <- function(
  file,
  n_mels = 80L,
  device = "auto",
  dtype = "auto"
) {
  device <- parse_device(device)
  dtype <- parse_dtype(dtype, device)

  # Load audio if file path provided
  if (is.character(file)) {
    audio <- load_audio(file)
  } else {
    audio <- as.numeric(file)
  }

  # Pad or trim to 30 seconds
  audio <- pad_or_trim(audio)

  # Compute STFT
  stft <- compute_stft(audio)

  # Compute magnitude (power spectrum)
  magnitudes <- stft$abs()$pow(2L)

  # Get mel filterbank (pre-computed from Whisper)
  mel_fb <- load_mel_filterbank(n_mels = n_mels)
  mel_fb_tensor <- torch::torch_tensor(mel_fb, dtype = torch::torch_float())

  # Apply mel filterbank: (n_mels, n_freqs) @ (n_freqs, n_frames) -> (n_mels, n_frames)
  mel_spec <- torch::torch_matmul(mel_fb_tensor, magnitudes)

  # Clamp to avoid log(0)
  mel_spec <- torch::torch_clamp(mel_spec, min = 1e-10)

  # Log mel spectrogram
  log_mel <- mel_spec$log10()

  # Whisper normalization: clamp to max - 8.0 dB range
  max_val <- log_mel$max()
  log_mel <- torch::torch_maximum(log_mel, max_val - 8.0)
  log_mel <- (log_mel + 4.0) / 4.0

  # Add batch dimension: (1, n_mels, n_frames)
  log_mel <- log_mel$unsqueeze(1L)

  # Move to device and dtype
  log_mel$to(device = device, dtype = dtype)
}

#' Get Audio Duration
#'
#' @param file Path to audio file
#' @return Duration in seconds
audio_duration <- function(file) {
  # Suppress FFmpeg stderr messages ("Estimating duration from bitrate")
  invisible(capture.output(
      info <- av::av_media_info(file),
      type = "message"
    ))
  info$duration
}

#' Split Long Audio into Chunks
#'
#' Split audio longer than 30 seconds into overlapping chunks.
#'
#' @param file Path to audio file
#' @param chunk_length Chunk length in seconds
#' @param overlap Overlap between chunks in seconds
#' @return List of audio chunks (numeric vectors)
split_audio <- function(
  file,
  chunk_length = 30,
  overlap = 1
) {
  audio <- load_audio(file)
  n_samples <- length(audio)

  chunk_samples <- as.integer(chunk_length * WHISPER_SAMPLE_RATE)
  overlap_samples <- as.integer(overlap * WHISPER_SAMPLE_RATE)
  hop_samples <- chunk_samples - overlap_samples

  chunks <- list()
  start <- 1L

  while (start <= n_samples) {
    end <- min(start + chunk_samples - 1L, n_samples)
    chunk <- audio[start:end]

    # Pad if necessary
    if (length(chunk) < chunk_samples) {
      chunk <- c(chunk, rep(0, chunk_samples - length(chunk)))
    }

    chunks <- c(chunks, list(chunk))
    start <- start + hop_samples

    # Stop if we've processed all audio
    if (end >= n_samples) break
  }

  chunks
}

