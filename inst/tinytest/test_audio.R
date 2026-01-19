# Tests for audio preprocessing

# Test mel filterbank creation
fb <- whisper:::create_mel_filterbank_fallback(n_mels = 80L)
expect_equal(nrow(fb), 80)
expect_equal(ncol(fb), 201) # n_fft/2 + 1

# Test Hz to Mel conversion
expect_equal(whisper:::hz_to_mel(0), 0)
expect_true(whisper:::hz_to_mel(1000) > 0)

# Test Mel to Hz conversion (inverse)
hz <- 440
mel <- whisper:::hz_to_mel(hz)
hz_back <- whisper:::mel_to_hz(mel)
expect_equal(hz_back, hz, tolerance = 0.01)

# Test pad_or_trim
audio <- 1:100
padded <- whisper:::pad_or_trim(audio, 200)
expect_equal(length(padded), 200)
expect_equal(padded[1:100], audio)
expect_equal(padded[101:200], rep(0, 100))

trimmed <- whisper:::pad_or_trim(audio, 50)
expect_equal(length(trimmed), 50)
expect_equal(trimmed, 1:50)

