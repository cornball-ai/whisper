# whisper streaming + endpointing design

Status: APPROVED 2026-08-25 (Troy). Decisions: Silero is a download, not
bundled (Q1); punctuation-aware endpointing is in v1 (Q2); one live
stream per process (Q3). Implementation on the `stream` branch.

Implements the capability behind `SpeechToText.Transcribe` in
`cornball-ai/fluffychat/proto/gpu_voice.proto` (merged in fluffychat PR #10).
The proto's field comments are the contract; this doc maps them onto whisper.

## Scope

whisper grows an in-process streaming session API: PCM chunks in, transcript
events out. That's the whole surface.

Out of scope, by design:

- **No gRPC, no sockets, no auth.** gpu.ctl owns the wire and the
  allocation-bearer validation; whisper stays auth-unaware. The
  whisper-to-gpu.ctl process mapping is a separate conversation with vientito.
- **No barge-in, no conversation state.** The client's energy detector decides
  barge-in; corteza owns dialogue. whisper only judges "the speaker's turn is
  over."
- **No resampling in v1.** The client contract sends 16 kHz mono S16LE. A
  session constructed with any other rate/channel count errors immediately
  (the bridge maps that to INVALID_ARGUMENT). Add resampling only when a real
  platform can't produce 16 kHz.

## Public API

```r
stream <- whisper_stream(pipe, language = NULL, ...tunables...)

events <- stream$feed(pcm)   # raw vector (S16LE bytes) or integer vector
events <- stream$end()       # end of stream: flush any open turn
stream$close()               # free buffers / VAD state
```

`feed()` is synchronous and single-threaded, matching `serve()`'s design: the
caller hands over whatever frame the platform produced (size explicitly not
constant), and gets back a list of zero or more events, in order:

```r
list(type = "transcript", text = "...", stable = TRUE|FALSE)
list(type = "speech_ended", audio_offset_ms = 12345)
```

Event semantics are the proto's, verbatim:

- **stable appends**: the turn's text is the concatenation of stable events in
  arrival order, whitespace included by us, nothing inserted by the client, no
  restating.
- **provisional replaces**: each provisional is the whole uncommitted tail;
  the next one supersedes it.
- **flush before SpeechEnded**: every token that will ever be stable for this
  turn is emitted as stable *before* the `speech_ended` event. The client
  discards leftover provisional; an endpointer that fires before the tail is
  transcribed eats the user's last words.

One session = one proto stream = many turns. After `speech_ended` the session
keeps consuming audio for the next turn; per-turn state resets internally, the
stream sample counter does not.

The stream clock is the cumulative sample count (`samples / 16` = ms). Never
per-message sizes — framing is not constant.

## Architecture

Three layers inside `feed()`:

```
PCM chunk
  └─ ingest: S16LE -> float [-1,1], append to ring buffer, advance clock
       └─ VAD: per-frame speech probability + turn state machine
            └─ decode: incremental re-decode of the open utterance
                 └─ events out
```

### Turn state machine

```
LISTENING --(speech >= min_speech_ms)--> IN_SPEECH
IN_SPEECH --(trailing non-speech >= endpoint_silence_ms)--> FLUSH -> LISTENING
```

- Entering IN_SPEECH, the utterance buffer starts `pre_roll_ms` (default 320)
  before the detected onset, so plosive-led first words aren't clipped.
- `audio_offset_ms` on `speech_ended` = end of the last VAD speech frame (the
  turn ends when the human stops talking, not when we finish waiting).

### VAD: Silero, with an energy fallback

Primary: **Silero VAD** (MIT), the TorchScript `silero_vad.jit` loaded via
`torch::jit_load()`. ~2 MB, CPU, ~1 ms per 32 ms frame. No new package
dependency — torch is already an Import; the model file is a download into the
existing cache, fetched like the whisper weights (download.R grows a fetcher).

Why not energy: the contract's own rationale — a breath and a sentence end
look identical to an energy threshold — applies server-side too, and the input
will carry AEC residue of the assistant's own synthesized speech during
replies. A trained VAD discriminates both; an energy gate does neither.

Whisper's own `no_speech_prob` is not the endpointer: it's computed per decode
window, far too coarse and too expensive for a 700 ms silence decision. It
stays as the existing hallucination gate inside decode.

Tunables (defaults, all `whisper_stream()` args):

| arg | default | meaning |
|---|---|---|
| `onset_prob` | 0.50 | speech-prob to enter speech |
| `offset_prob` | 0.35 | hysteresis threshold to leave speech |
| `min_speech_ms` | 250 | shorter blips never open a turn |
| `endpoint_silence_ms` | 700 | trailing non-speech that ends a turn |
| `pre_roll_ms` | 320 | audio kept from before onset |

**Spike risk (gate on milestone 0):** R torch's `jit_load()` must actually run
the Silero archive. Validate before anything else is built. If it can't,
fallback is an energy + spectral-flatness VAD in base R behind the same
interface (`vad = "energy"`), with honest documentation that its endpointing
is weaker; the interface doesn't change either way.

### Streaming decode: LocalAgreement-2 with prefix conditioning

Whisper is a 30 s window batch model; the streaming recipe is re-decoding a
growing buffer and committing what consecutive decodes agree on
(LocalAgreement-2, Macháček et al. 2023). It maps one-to-one onto the proto:
the agreed prefix is `stable`, the disagreeing tail is `provisional`.

Per decode tick (every `decode_interval_ms` = 1000 of *new* audio, and only
when the previous tick has finished — cadence degrades gracefully when decode
is slower than realtime):

1. Mel of the utterance buffer (`audio_to_mel()` already accepts raw sample
   vectors), encode, greedy decode via the existing jit path. Temperature 0
   only — the fallback ladder is a latency cost that provisional text doesn't
   justify.
2. **Prefix conditioning:** the decoder's starting tokens are
   `get_initial_tokens(...)` + all committed content tokens. Committed text is
   forced, so no decode can ever contradict what was already emitted as
   stable. (Mechanically: `greedy_decode()` already takes the starting token
   tensor; committed tokens ride after the SOT sequence. KV cache makes the
   forced prefix one batched forward pass.)
3. Longest common token prefix of this hypothesis vs. the previous tick's
   (beyond the committed point) → emit as one `stable` event, decoded to text
   over exactly that token range (BPE carries the joining whitespace, so
   append-concatenation is byte-correct).
4. Remaining tail → one `provisional` event (whole tail, replaces).

Commit at token level, not text level: the committed token vector is both the
append-exactness guarantee and the forced prefix for the next tick.

### Endpoint flush (ordering is the contract)

When the state machine fires:

1. Final decode of the full utterance buffer, prefix-conditioned on committed
   tokens. This is the one decode allowed quality options (it feeds the
   agent): quality-gated retry per the batch path's thresholds, configurable
   `final_beam_size` (default 1).
2. Emit the entire uncommitted remainder as **one stable event**.
3. Emit `speech_ended` with `audio_offset_ms`.
4. Reset per-turn state (buffer, hypotheses, committed tokens); stream clock
   and VAD keep running.

Nothing provisional is ever emitted between steps 2 and 3.

### Long utterances (> ~28 s)

Rare in conversation, must not corrupt when it happens. When the open
utterance buffer reaches `max_utterance_s` (28):

- Run a decode with timestamps, hard-commit all segments whose end timestamp
  is at least 5 s behind the buffer end (emit as stable).
- Advance the buffer start to the last committed timestamp, keep decoding in
  the new window with the committed tail as prefix context.

This is the batch seek loop's logic applied incrementally. No `speech_ended`
fires — the turn is still open; only the window slid.

### Language

Detect once, on the first decode tick of the stream's first utterance, then
pin for the session (`language = NULL` semantics). An explicit `language`
skips detection. Per-tick detection would jitter the transcript and waste a
forward pass. Matches the proto: language is a stream-config field, not
per-turn.

## Files

One responsibility per file, all new:

- `R/vad.R` — Silero jit load + download, energy fallback, frame loop, turn
  state machine. Pure function core (prob sequence in, state transitions out)
  so it tests without audio.
- `R/stream.R` — `whisper_stream()`: session object, ingest/clock/ring
  buffer, event assembly, flush sequence.
- `R/stream_decode.R` — LocalAgreement hypothesis tracking, prefix-conditioned
  tick decode, token-level commit bookkeeping.

Touched: `R/download.R` (VAD model fetch), NAMESPACE/docs via tinyrox.

## Tests (tinytest)

The fluffychat client has an executable spec
(`live_voice_session_test.dart`, "provisional transcripts replace; stable
ones append"); mirror it from the producing side:

- **Contract unit tests, no model:** synthetic hypothesis sequences through
  the LocalAgreement core — stable events never restate, concatenation equals
  final text exactly (whitespace included), provisional is always the full
  tail, no provisional between final stable and speech_ended.
- **VAD state machine, no model:** synthetic prob sequences — hysteresis,
  min-speech, pre-roll, endpoint timing, offset arithmetic.
- **Integration (`at_home()`):** feed `inst/audio/jfk.mp3` as S16LE chunks of
  random sizes (the not-constant framing rule, exercised); assert the stable
  concatenation matches batch `transcribe()` output, exactly one
  `speech_ended`, sane offset. Chunk-size randomization seeded.
- **AEC-residue corpus (milestone 4):** speech mixed with attenuated
  synthesized speech (chatterbox output at low gain) — endpointer must not
  hold the turn open on residue.

## Milestones

0. **Spike (gate):** `jit_load()` the Silero archive in R torch, sane probs on
   jfk.mp3 vs. silence. Everything else waits on this answer.
1. `R/vad.R` + state machine + unit tests.
2. `R/stream.R` + `R/stream_decode.R`: LocalAgreement, events, contract tests.
3. Endpoint flush + `speech_ended` + integration test vs. batch output.
4. Overflow handling + AEC-residue corpus.
5. Dev harness (`simulate_stream(file, chunk_jitter)`) + docs; hand the
   in-process API shape to vientito for the gpu.ctl bridge.

Feature work on a branch; this is a minor-version feature (0.6.0) when it
ships. `^tasks$` needs adding to `.Rbuildignore` before any build happens with
this file present.

## Latency expectation (large-v3, resident GPU)

- Provisional cadence: ~1 s behind speech (decode interval + decode time).
- Stable lag: one extra tick behind provisional (LocalAgreement-2 needs two
  agreeing decodes).
- `speech_ended` after last word: `endpoint_silence_ms` (700) + final decode
  (~0.5 s) ≈ 1.2 s. Smaller models tighten this; model choice is the host's
  (`TranscribeConfig.model`, empty = resident default).

## Open questions for Troy

1. **Silero VAD dependency-by-download** — a second model artifact fetched at
   first use (MIT license, ~2 MB, cached like the weights). OK, or bundle it
   in inst/ (tarball weight vs. no first-run download)?
2. **Punctuation-aware endpointing** (v1.5): shorten `endpoint_silence_ms` to
   ~500 when the provisional tail ends in terminal punctuation, stretch to
   ~1000 mid-sentence. Cheap to add on top of this design; worth it, or keep
   the fixed timer?
3. **Concurrency:** one live stream per process in v1 (single GPU, same
   stance as `serve()`). Multi-session needs a second process or serialized
   decode ticks — acceptable for now?
