/**
 * Frontend policy aligned with Go SPEECH_MAX_UPLOAD_BYTES default (10 MiB
 * whole HTTP body) and C0 SPEECH_MAX_AUDIO_SECONDS (30).
 *
 * Go MaxAudioBytes subtracts a 64 KiB multipart reserve so the re-encoded
 * upstream body stays under the FastAPI cap. The browser uses the same WAV
 * payload ceiling. Deployments may lower SPEECH_MAX_UPLOAD_BYTES; Go remains
 * the authority and this file is a hard client-side ceiling, not a per-env
 * probe.
 */
export const SPEECH_SAMPLE_RATE = 16000
export const SPEECH_CHANNELS = 1
export const SPEECH_BITS_PER_SAMPLE = 16
export const SPEECH_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
export const SPEECH_MULTIPART_OVERHEAD_BYTES = 64 * 1024
export const SPEECH_MAX_WAV_BYTES = SPEECH_MAX_UPLOAD_BYTES - SPEECH_MULTIPART_OVERHEAD_BYTES
export const SPEECH_MAX_AUDIO_SECONDS = 30
export const SPEECH_MIN_AUDIO_SECONDS = 0.2
export const SPEECH_TIMEOUT_MS = 130000
export const SPEECH_MODEL_ID = 'c0_cascaded'
export const SPEECH_FIELD = 'audio'
export const SPEECH_PATH = (sessionId) =>
  `/api/v1/sessions/${encodeURIComponent(sessionId)}/speech/translate`
