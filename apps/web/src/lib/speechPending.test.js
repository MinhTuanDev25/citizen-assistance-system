import { describe, expect, it } from 'vitest'
import { isCurrentSpeechPending, makeSpeechPending } from './speechPending.js'

describe('speech pending contract', () => {
  it('accepts microphone audio only for the current generation', () => {
    const pending = makeSpeechPending(
      { bytes: new Uint8Array([1, 2, 3]), durationMs: 200, sampleCount: 3200 },
      { name: 'microphone.wav', source: 'microphone', gen: 3 },
    )
    expect(pending.gen).toBe(3)
    expect(pending.source).toBe('microphone')
    expect(isCurrentSpeechPending(pending, 3)).toBe(true)
    expect(isCurrentSpeechPending(pending, 4)).toBe(false)
  })

  it('rejects encoded samples that never received a generation stamp', () => {
    const encoded = { bytes: new Uint8Array([1]), durationMs: 200, sampleCount: 3200 }
    expect(isCurrentSpeechPending(encoded, 0)).toBe(false)
    expect(isCurrentSpeechPending({ ...encoded, gen: undefined }, 0)).toBe(false)
  })
})
