import { describe, expect, it } from 'vitest'
import {
  SPEECH_MAX_AUDIO_SECONDS,
  SPEECH_MAX_WAV_BYTES,
  SPEECH_SAMPLE_RATE,
} from '../config/speech.js'
import {
  assertWavUploadSize,
  concatFloat32,
  encodeCapturedSamples,
  encodeWavPcm16Mono,
  maxCaptureSamples,
  mixToMono,
  parseWav,
  resampleTo,
  validateBahnarWav,
  wavDurationSeconds,
} from './wav.js'

function tone(n, freq = 440, rate = SPEECH_SAMPLE_RATE) {
  const samples = new Float32Array(n)
  for (let i = 0; i < n; i += 1) {
    samples[i] = Math.sin((2 * Math.PI * freq * i) / rate) * 0.5
  }
  return samples
}

function rms(samples) {
  let sum = 0
  for (let i = 0; i < samples.length; i += 1) sum += samples[i] * samples[i]
  return Math.sqrt(sum / samples.length)
}

function writeAscii(bytes, offset, text) {
  for (let i = 0; i < text.length; i += 1) bytes[offset + i] = text.charCodeAt(i)
}

function craftWav({
  channels = 1,
  sampleRate = 16000,
  bits = 16,
  samples = 1600,
  format = 1,
  riffSize,
  dataSize,
  extraBeforeData,
} = {}) {
  const block = channels * (bits / 8)
  const payload = (dataSize != null ? dataSize : samples * block)
  const extras = extraBeforeData || new Uint8Array(0)
  const bytes = new Uint8Array(12 + 8 + 16 + extras.length + 8 + payload)
  const view = new DataView(bytes.buffer)
  writeAscii(bytes, 0, 'RIFF')
  view.setUint32(4, riffSize != null ? riffSize : bytes.length - 8, true)
  writeAscii(bytes, 8, 'WAVE')
  writeAscii(bytes, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, format, true)
  view.setUint16(22, channels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * block, true)
  view.setUint16(32, block, true)
  view.setUint16(34, bits, true)
  let offset = 36
  if (extras.length) {
    bytes.set(extras, offset)
    offset += extras.length
  }
  writeAscii(bytes, offset, 'data')
  view.setUint32(offset + 4, payload, true)
  return bytes
}

function listChunk(size, pad = true) {
  const body = size + (pad && size % 2 === 1 ? 1 : 0)
  const bytes = new Uint8Array(8 + body)
  const view = new DataView(bytes.buffer)
  writeAscii(bytes, 0, 'LIST')
  view.setUint32(4, size, true)
  return bytes
}

describe('WAV PCM16 mono 16 kHz', () => {
  it('encodes a valid RIFF/WAVE with correct sizes', () => {
    const samples = tone(16000)
    const bytes = encodeWavPcm16Mono(samples)
    expect(String.fromCharCode(...bytes.slice(0, 4))).toBe('RIFF')
    expect(String.fromCharCode(...bytes.slice(8, 12))).toBe('WAVE')
    const parsed = parseWav(bytes)
    expect(parsed.audioFormat).toBe(1)
    expect(parsed.channels).toBe(1)
    expect(parsed.sampleRate).toBe(16000)
    expect(parsed.bitsPerSample).toBe(16)
    expect(parsed.dataSize).toBe(16000 * 2)
    expect(bytes.byteLength).toBe(44 + 16000 * 2)
    expect(wavDurationSeconds(parsed)).toBeCloseTo(1, 5)
    expect(validateBahnarWav(bytes).ok).toBe(true)
  })

  it('accepts a valid WAV with an extra LIST metadata chunk', () => {
    const extra = listChunk(4)
    const bytes = craftWav({ samples: 16000, extraBeforeData: extra })
    expect(validateBahnarWav(bytes).ok).toBe(true)
    expect(parseWav(bytes).dataOffset).toBeGreaterThan(44)
  })

  it('rejects empty input', () => {
    expect(() => encodeWavPcm16Mono(new Float32Array(0))).toThrow(/trống/i)
    expect(validateBahnarWav(new Uint8Array()).ok).toBe(false)
    expect(() => encodeCapturedSamples([], 48000)).toThrow(/trống/i)
  })

  it('rejects the previously accepted truncated WAV whose header still declares 32k audio', () => {
    const full = encodeWavPcm16Mono(tone(16000))
    expect(full.byteLength).toBe(32044)
    expect(validateBahnarWav(full).ok).toBe(true)
    const truncated = full.slice(0, 100)
    const check = validateBahnarWav(truncated)
    expect(check.ok).toBe(false)
    expect(check.error).toMatch(/cắt|vượt quá/i)
  })

  it('rejects truncated RIFF header, fmt, and data payloads', () => {
    expect(validateBahnarWav(new Uint8Array(8)).ok).toBe(false)
    const fmtCut = craftWav({ samples: 16000 }).slice(0, 28)
    expect(validateBahnarWav(fmtCut).ok).toBe(false)
    const dataCut = craftWav({ samples: 16000, dataSize: 32000 }).slice(0, 80)
    expect(validateBahnarWav(dataCut).ok).toBe(false)
  })

  it('rejects declared dataSize larger than remaining bytes', () => {
    const bytes = craftWav({ samples: 1600 })
    new DataView(bytes.buffer).setUint32(40, 32000, true)
    expect(validateBahnarWav(bytes).ok).toBe(false)
  })

  it('rejects incorrect RIFF size and out-of-bounds chunks', () => {
    const smallRiff = craftWav({ samples: 16000, riffSize: 12 })
    expect(validateBahnarWav(smallRiff).ok).toBe(false)
    const overflow = craftWav({ samples: 1600 })
    const view = new DataView(overflow.buffer)
    view.setUint32(16, 0xfffffff0, true)
    expect(validateBahnarWav(overflow).ok).toBe(false)
  })

  it('rejects empty audio payload and inconsistent block alignment', () => {
    expect(validateBahnarWav(craftWav({ samples: 0, dataSize: 0 })).ok).toBe(false)
    const misaligned = craftWav({ samples: 16000 })
    const view = new DataView(misaligned.buffer)
    view.setUint16(32, 3, true)
    expect(validateBahnarWav(misaligned).ok).toBe(false)
  })

  it('parses odd-sized chunks that include a pad byte', () => {
    const extra = listChunk(1, true)
    const bytes = craftWav({ samples: 16000, extraBeforeData: extra })
    expect(validateBahnarWav(bytes).ok).toBe(true)
  })

  it('rejects dangling bytes inside the declared RIFF size', () => {
    const base = encodeWavPcm16Mono(tone(16000))
    for (let n = 1; n <= 7; n += 1) {
      const out = new Uint8Array(base.length + n)
      out.set(base)
      new DataView(out.buffer).setUint32(4, base.length - 8 + n, true)
      expect(validateBahnarWav(out).ok, `${n} dangling`).toBe(false)
    }
  })

  it('rejects an odd-sized LIST chunk missing its pad byte', () => {
    const extra = listChunk(1, false)
    const bytes = craftWav({ samples: 16000, extraBeforeData: extra })
    expect(validateBahnarWav(bytes).ok).toBe(false)
  })

  it('accepts valid LIST metadata after the data chunk', () => {
    const base = encodeWavPcm16Mono(tone(16000))
    const extra = listChunk(4)
    const out = new Uint8Array(base.length + extra.length)
    out.set(base)
    out.set(extra, base.length)
    new DataView(out.buffer).setUint32(4, out.length - 8, true)
    expect(validateBahnarWav(out).ok).toBe(true)
  })

  it('rejects malformed metadata after a valid data chunk', () => {
    const base = encodeWavPcm16Mono(tone(16000))
    const extra = listChunk(4)
    const out = new Uint8Array(base.length + extra.length)
    out.set(base)
    out.set(extra, base.length)
    new DataView(out.buffer).setUint32(4, out.length - 8, true)
    new DataView(out.buffer).setUint32(base.length + 4, 0xffffff00, true)
    expect(validateBahnarWav(out).ok).toBe(false)
  })

  it('allows bytes after the declared RIFF end, matching Go prefix-only checks', () => {
    const base = encodeWavPcm16Mono(tone(16000))
    const out = new Uint8Array(base.length + 4)
    out.set(base)
    expect(validateBahnarWav(out).ok).toBe(true)
  })

  it('rejects stereo, 8 kHz, non-PCM, truncated, and oversized-duration files', () => {
    expect(validateBahnarWav(craftWav({ channels: 2 })).ok).toBe(false)
    expect(validateBahnarWav(craftWav({ sampleRate: 8000 })).ok).toBe(false)
    expect(validateBahnarWav(craftWav({ format: 3 })).ok).toBe(false)
    expect(validateBahnarWav(craftWav({ bits: 8, samples: 16000 })).ok).toBe(false)
    expect(validateBahnarWav(new Uint8Array([73, 68, 51, 4, 5])).ok).toBe(false)
    const tooShort = craftWav({ samples: 100 })
    expect(validateBahnarWav(tooShort).error).toMatch(/ngắn/)
    const tooLong = craftWav({ samples: (SPEECH_MAX_AUDIO_SECONDS + 1) * 16000 })
    expect(validateBahnarWav(tooLong).ok).toBe(false)
  })

  it('rejects files above the WAV payload ceiling', () => {
    const bytes = craftWav({ samples: 16000 })
    expect(validateBahnarWav(bytes, { maxBytes: 20 }).ok).toBe(false)
    expect(assertWavUploadSize({ size: SPEECH_MAX_WAV_BYTES + 1 }).ok).toBe(false)
    expect(assertWavUploadSize({ size: 0 }).ok).toBe(false)
    expect(assertWavUploadSize({ size: SPEECH_MAX_WAV_BYTES }).ok).toBe(true)
  })

  it('concatenates chunks without copying holes', () => {
    const a = new Float32Array([1, 2])
    const b = new Float32Array([3])
    expect(Array.from(concatFloat32([a, b]))).toEqual([1, 2, 3])
  })

  it('mixes stereo buffers to mono', () => {
    const left = new Float32Array([1, 0])
    const right = new Float32Array([0, 1])
    expect(Array.from(mixToMono([left, right]))).toEqual([0.5, 0.5])
  })
})

describe('windowed-sinc resample anti-aliasing', () => {
  const PASSBAND_RMS_MIN = 0.28
  const PASSBAND_RMS_MAX = 0.42
  const STOPBAND_RMS_MAX = 0.05
  const expectedSineRms = 0.5 / Math.SQRT2

  it('keeps 1 kHz passband energy from 48 kHz and 44.1 kHz', () => {
    const from48 = resampleTo(tone(48000, 1000, 48000), 48000, 16000)
    const from44 = resampleTo(tone(44100, 1000, 44100), 44100, 16000)
    expect(from48.length).toBe(16000)
    expect(from44.length).toBe(16000)
    const rms48 = rms(from48)
    const rms44 = rms(from44)
    expect(rms48).toBeGreaterThan(PASSBAND_RMS_MIN)
    expect(rms48).toBeLessThan(PASSBAND_RMS_MAX)
    expect(rms44).toBeGreaterThan(PASSBAND_RMS_MIN)
    expect(rms44).toBeLessThan(PASSBAND_RMS_MAX)
    expect(Math.abs(rms48 - expectedSineRms)).toBeLessThan(0.08)
  })

  it('keeps 3 kHz speech-band energy', () => {
    const out = resampleTo(tone(48000, 3000, 48000), 48000, 16000)
    expect(rms(out)).toBeGreaterThan(PASSBAND_RMS_MIN)
  })

  it('attenuates 12 kHz above the 8 kHz Nyquist from 48 kHz and 44.1 kHz', () => {
    const from48 = resampleTo(tone(48000, 12000, 48000), 48000, 16000)
    const from44 = resampleTo(tone(44100, 12000, 44100), 44100, 16000)
    const rms48 = rms(from48)
    const rms44 = rms(from44)
    expect(rms48).toBeLessThan(STOPBAND_RMS_MAX)
    expect(rms44).toBeLessThan(STOPBAND_RMS_MAX)
    expect(rms48).toBeLessThan(0.1)
    expect(rms44).toBeLessThan(0.1)
  })

  it('does not reject silence or near-silence as invalid WAV (no RMS/VAD gate)', () => {
    const silence = encodeWavPcm16Mono(new Float32Array(16000))
    expect(validateBahnarWav(silence).ok).toBe(true)
    const nearSilence = encodeWavPcm16Mono(new Float32Array(16000).fill(0.0008))
    expect(validateBahnarWav(nearSilence).ok).toBe(true)
    const speechLike = encodeWavPcm16Mono(tone(16000))
    expect(validateBahnarWav(speechLike).ok).toBe(true)
  })

  it('preserves duration, silence, and short recordings', () => {
    const oneSec = resampleTo(tone(48000, 200, 48000), 48000, 16000)
    expect(oneSec.length / 16000).toBeCloseTo(1, 5)
    const silence = resampleTo(new Float32Array(48000), 48000, 16000)
    expect(rms(silence)).toBeLessThan(1e-7)
    const short = resampleTo(tone(9600, 400, 48000), 48000, 16000)
    expect(short.length).toBe(3200)
    const encoded = encodeCapturedSamples([tone(48000, 200, 48000)], 48000)
    expect(encoded.sampleCount).toBe(16000)
    expect(encoded.durationMs).toBe(1000)
    expect(parseWav(encoded.bytes).sampleRate).toBe(16000)
    expect(validateBahnarWav(encoded.bytes).ok).toBe(true)
  })

  it('downsamples a 48 kHz integer-ratio buffer to 16 kHz length', () => {
    const input = new Float32Array(6).fill(1)
    const out = resampleTo(input, 48000, 16000)
    expect(out.length).toBe(2)
  })
})

describe('microphone duration clamp', () => {
  it('accepts 30.03s capture audio by clipping to 30s', () => {
    const over = Math.round(16000 * 30.03)
    const encoded = encodeCapturedSamples([tone(over)], 16000)
    expect(encoded.sampleCount).toBe(maxCaptureSamples(16000))
    expect(encoded.durationMs).toBe(SPEECH_MAX_AUDIO_SECONDS * 1000)
    expect(validateBahnarWav(encoded.bytes).ok).toBe(true)
  })

  it('clips 30.06s and 48 kHz overrun instead of rejecting', () => {
    const over16 = Math.round(16000 * 30.06)
    const encoded16 = encodeCapturedSamples([tone(over16)], 16000)
    expect(encoded16.sampleCount).toBe(480000)
    expect(encoded16.durationMs).toBe(30000)

    const over48 = Math.round(48000 * 30.2)
    const encoded48 = encodeCapturedSamples([tone(over48, 200, 48000)], 48000)
    expect(encoded48.sampleCount).toBe(480000)
    expect(encoded48.durationMs).toBe(30000)
    expect(validateBahnarWav(encoded48.bytes).ok).toBe(true)
  })

  it('still rejects an uploaded WAV that is longer than 30.05s', () => {
    const tooLong = encodeWavPcm16Mono(tone(Math.round(16000 * 30.06)))
    const check = validateBahnarWav(tooLong)
    expect(check.ok).toBe(false)
    expect(check.error).toMatch(/vượt quá 30/)
  })
})
