/**
 * Shared pending-audio contract for Microphone and Upload.
 * `gen` must match BahnarSpeechPanel's fileGenRef at translate time.
 */
export function makeSpeechPending(encoded, { name, source, gen }) {
  return {
    bytes: encoded.bytes,
    durationMs: encoded.durationMs,
    sampleCount: encoded.sampleCount,
    name,
    source,
    gen,
  }
}

export function isCurrentSpeechPending(pending, gen) {
  return Boolean(
    pending?.bytes &&
      pending.gen === gen &&
      (pending.source === 'microphone' || pending.source === 'upload'),
  )
}
