import {
  SPEECH_BITS_PER_SAMPLE,
  SPEECH_CHANNELS,
  SPEECH_MAX_AUDIO_SECONDS,
  SPEECH_MAX_WAV_BYTES,
  SPEECH_MIN_AUDIO_SECONDS,
  SPEECH_SAMPLE_RATE,
} from '../config/speech.js'

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i))
  }
}

export function concatFloat32(chunks) {
  const list = (chunks || []).filter((c) => c && c.length)
  if (!list.length) return new Float32Array(0)
  let total = 0
  for (const c of list) total += c.length
  const out = new Float32Array(total)
  let offset = 0
  for (const c of list) {
    out.set(c, offset)
    offset += c.length
  }
  return out
}

export function mixToMono(channels) {
  if (!channels?.length) return new Float32Array(0)
  if (channels.length === 1) return Float32Array.from(channels[0])
  const len = channels[0].length
  const out = new Float32Array(len)
  const n = channels.length
  for (let i = 0; i < len; i += 1) {
    let sum = 0
    for (let c = 0; c < n; c += 1) {
      sum += channels[c][i] || 0
    }
    out[i] = sum / n
  }
  return out
}

const SINC_HALF_WIDTH = 32

function sinc(x) {
  if (x === 0) return 1
  const p = Math.PI * x
  return Math.sin(p) / p
}

function hamming(i, n) {
  if (n <= 0) return 1
  return 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / n)
}

/**
 * Windowed-sinc resampler with a low-pass cutoff at 0.9 × min(from,to)/2.
 * Deterministic; no AudioContext. Typical capture rates 44.1/48 kHz → 16 kHz.
 */
export function resampleTo(samples, fromRate, toRate) {
  if (!samples?.length) return new Float32Array(0)
  if (!fromRate || fromRate <= 0 || !toRate || toRate <= 0) {
    throw new Error('Tần số lấy mẫu không hợp lệ.')
  }
  if (fromRate === toRate) return Float32Array.from(samples)
  const outLen = Math.max(1, Math.round((samples.length * toRate) / fromRate))
  const out = new Float32Array(outLen)
  const fc = (0.9 * Math.min(fromRate, toRate) * 0.5) / fromRate
  const half = SINC_HALF_WIDTH
  const winN = half * 2
  const last = samples.length - 1
  for (let i = 0; i < outLen; i += 1) {
    const src = (i * fromRate) / toRate
    const i0 = Math.floor(src) - half
    let sum = 0
    let wsum = 0
    for (let k = i0; k <= i0 + winN; k += 1) {
      const t = k - src
      const w = sinc(2 * fc * t) * hamming(k - i0, winN)
      const sample = k < 0 || k > last ? 0 : samples[k]
      sum += sample * w
      wsum += w
    }
    out[i] = wsum !== 0 ? sum / wsum : 0
  }
  return out
}

export function floatToPcm16(samples) {
  const out = new Int16Array(samples.length)
  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i] || 0))
    out[i] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff)
  }
  return out
}

export function encodeWavPcm16Mono(samples, sampleRate = SPEECH_SAMPLE_RATE) {
  if (!samples?.length) {
    throw new Error('Bản ghi trống.')
  }
  if (sampleRate !== SPEECH_SAMPLE_RATE) {
    throw new Error('WAV phải là 16 kHz.')
  }
  const pcm = samples instanceof Int16Array ? samples : floatToPcm16(samples)
  const dataSize = pcm.length * 2
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)
  writeAscii(view, 0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true)
  writeAscii(view, 8, 'WAVE')
  writeAscii(view, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, SPEECH_CHANNELS, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * SPEECH_CHANNELS * (SPEECH_BITS_PER_SAMPLE / 8), true)
  view.setUint16(32, SPEECH_CHANNELS * (SPEECH_BITS_PER_SAMPLE / 8), true)
  view.setUint16(34, SPEECH_BITS_PER_SAMPLE, true)
  writeAscii(view, 36, 'data')
  view.setUint32(40, dataSize, true)
  let offset = 44
  for (let i = 0; i < pcm.length; i += 1) {
    view.setInt16(offset, pcm[i], true)
    offset += 2
  }
  return new Uint8Array(buffer)
}

function readFourCC(bytes, offset) {
  return String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3])
}

export function parseWav(bytes) {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes || [])
  if (buf.length < 12) {
    throw new Error('File WAV không hợp lệ hoặc bị cắt.')
  }
  if (readFourCC(buf, 0) !== 'RIFF' || readFourCC(buf, 8) !== 'WAVE') {
    throw new Error('Không phải file WAV (RIFF/WAVE).')
  }
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  const riffSize = view.getUint32(4, true)
  if (riffSize < 4) {
    throw new Error('Kích thước RIFF không hợp lệ.')
  }
  const riffEnd = 8 + riffSize
  if (riffEnd > buf.length) {
    throw new Error('File WAV không hợp lệ hoặc bị cắt.')
  }
  let offset = 12
  let fmt = null
  let dataOffset = -1
  let dataSize = 0
  while (offset + 8 <= riffEnd) {
    const id = readFourCC(buf, offset)
    const size = view.getUint32(offset + 4, true)
    const body = offset + 8
    const remaining = riffEnd - body
    if (size > remaining) {
      throw new Error('Khối WAV vượt quá kích thước file.')
    }
    if (body + size > buf.length) {
      throw new Error('File WAV không hợp lệ hoặc bị cắt.')
    }
    let pad = 0
    if (size % 2 === 1) {
      if (body + size + 1 > riffEnd) {
        throw new Error('Khối WAV thiếu byte padding.')
      }
      pad = 1
    }
    if (id === 'fmt ') {
      if (size < 16) {
        throw new Error('Khối fmt bị cắt.')
      }
      const fmtView = new DataView(buf.buffer, buf.byteOffset + body, 16)
      fmt = {
        audioFormat: fmtView.getUint16(0, true),
        channels: fmtView.getUint16(2, true),
        sampleRate: fmtView.getUint32(4, true),
        byteRate: fmtView.getUint32(8, true),
        blockAlign: fmtView.getUint16(12, true),
        bitsPerSample: fmtView.getUint16(14, true),
      }
    } else if (id === 'data' && dataOffset < 0) {
      dataOffset = body
      dataSize = size
    }
    const next = body + size + pad
    if (next <= offset) {
      throw new Error('Kích thước khối WAV không hợp lệ.')
    }
    offset = next
  }
  if (offset !== riffEnd) {
    throw new Error('Cấu trúc RIFF không khép kín.')
  }
  if (!fmt || dataOffset < 0) {
    throw new Error('WAV thiếu khối fmt hoặc data.')
  }
  if (dataOffset + dataSize > buf.length || dataOffset + dataSize > riffEnd) {
    throw new Error('Khối dữ liệu WAV bị cắt.')
  }
  return { ...fmt, dataOffset, dataSize, byteLength: buf.length, riffSize }
}

export function wavDurationSeconds(parsed) {
  const bytesPerSample = (parsed.bitsPerSample / 8) * parsed.channels
  if (!bytesPerSample) return 0
  return parsed.dataSize / (parsed.sampleRate * bytesPerSample)
}

export function assertWavUploadSize(file, maxBytes = SPEECH_MAX_WAV_BYTES) {
  const size = Number(file?.size)
  if (!Number.isFinite(size) || size <= 0) {
    return { ok: false, error: 'File rỗng.' }
  }
  if (size > maxBytes) {
    return { ok: false, error: 'File vượt quá dung lượng cho phép.' }
  }
  return { ok: true, size }
}

export function validateBahnarWav(bytes, { maxBytes = SPEECH_MAX_WAV_BYTES } = {}) {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes || [])
  if (!buf.length) {
    return { ok: false, error: 'File rỗng.' }
  }
  if (buf.length > maxBytes) {
    return { ok: false, error: 'File vượt quá dung lượng cho phép.' }
  }
  let parsed
  try {
    parsed = parseWav(buf)
  } catch (err) {
    return { ok: false, error: err.message || 'File WAV không hợp lệ.' }
  }
  if (parsed.audioFormat !== 1) {
    return { ok: false, error: 'Chỉ nhận WAV PCM tuyến tính (không nén).' }
  }
  if (parsed.channels !== SPEECH_CHANNELS) {
    return { ok: false, error: 'Âm thanh phải là mono (1 kênh).' }
  }
  if (parsed.sampleRate !== SPEECH_SAMPLE_RATE) {
    return { ok: false, error: 'Tần số lấy mẫu phải là 16 kHz.' }
  }
  if (parsed.bitsPerSample !== SPEECH_BITS_PER_SAMPLE) {
    return { ok: false, error: 'Độ phân giải phải là PCM 16-bit.' }
  }
  const frame = SPEECH_CHANNELS * (SPEECH_BITS_PER_SAMPLE / 8)
  if (parsed.blockAlign !== frame) {
    return { ok: false, error: 'blockAlign WAV không khớp PCM 16-bit mono.' }
  }
  if (parsed.byteRate !== SPEECH_SAMPLE_RATE * frame) {
    return { ok: false, error: 'byteRate WAV không khớp 16 kHz PCM.' }
  }
  if (parsed.dataSize <= 0 || parsed.dataSize % frame !== 0) {
    return { ok: false, error: 'Khối dữ liệu WAV trống hoặc bị hỏng.' }
  }
  if (parsed.dataOffset + parsed.dataSize > parsed.byteLength) {
    return { ok: false, error: 'Khối dữ liệu WAV bị cắt.' }
  }
  const duration = wavDurationSeconds(parsed)
  if (duration < SPEECH_MIN_AUDIO_SECONDS) {
    return { ok: false, error: 'Bản ghi quá ngắn.' }
  }
  if (duration > SPEECH_MAX_AUDIO_SECONDS + 0.05) {
    return { ok: false, error: `Bản ghi vượt quá ${SPEECH_MAX_AUDIO_SECONDS} giây.` }
  }
  return { ok: true, parsed, duration, byteLength: buf.length }
}

export function maxCaptureSamples(sampleRate = SPEECH_SAMPLE_RATE) {
  return Math.floor(SPEECH_MAX_AUDIO_SECONDS * sampleRate)
}

function clipToMaxSeconds(samples, sampleRate) {
  const maxSamples = maxCaptureSamples(sampleRate)
  if (samples.length <= maxSamples) return samples
  return samples.subarray(0, maxSamples)
}

export function encodeCapturedSamples(chunks, inputSampleRate) {
  const raw = concatFloat32(chunks)
  if (!raw.length) {
    throw new Error('Bản ghi trống.')
  }
  const clippedInput = clipToMaxSeconds(raw, inputSampleRate)
  const resampled = resampleTo(clippedInput, inputSampleRate, SPEECH_SAMPLE_RATE)
  if (!resampled.length) {
    throw new Error('Bản ghi trống sau khi đổi tần số.')
  }
  const clipped = clipToMaxSeconds(resampled, SPEECH_SAMPLE_RATE)
  const duration = clipped.length / SPEECH_SAMPLE_RATE
  if (duration < SPEECH_MIN_AUDIO_SECONDS) {
    throw new Error('Bản ghi quá ngắn.')
  }
  const bytes = encodeWavPcm16Mono(clipped, SPEECH_SAMPLE_RATE)
  if (bytes.byteLength > SPEECH_MAX_WAV_BYTES) {
    throw new Error('File vượt quá dung lượng cho phép.')
  }
  return { bytes, durationMs: Math.round(duration * 1000), sampleCount: clipped.length }
}
