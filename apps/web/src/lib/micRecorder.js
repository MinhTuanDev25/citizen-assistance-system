import { SPEECH_MAX_AUDIO_SECONDS } from '../config/speech.js'
import { encodeCapturedSamples } from './wav.js'

function defaultGetUserMedia(constraints) {
  const md = navigator.mediaDevices
  if (!md?.getUserMedia) {
    const err = new Error('Trình duyệt không hỗ trợ micro. Dùng HTTPS hoặc localhost.')
    err.code = 'UNSUPPORTED'
    throw err
  }
  return md.getUserMedia(constraints)
}

function defaultAudioContext(opts) {
  const Ctor = window.AudioContext || window.webkitAudioContext
  if (!Ctor) {
    const err = new Error('Trình duyệt không hỗ trợ Web Audio.')
    err.code = 'UNSUPPORTED'
    throw err
  }
  return new Ctor(opts)
}

function stopTracks(stream) {
  if (!stream) return
  for (const track of stream.getTracks?.() || []) {
    try {
      track.stop()
    } catch {
      /* already stopped */
    }
  }
}

function wrapCaptureError(err, fallback) {
  if (err?.code === 'CAPTURE_CANCELLED' || err?.code === 'UNSUPPORTED') throw err
  const fail = new Error(err?.message || fallback)
  fail.code = err?.code || 'CAPTURE_FAILED'
  throw fail
}

/**
 * Capture live microphone samples and encode one 16 kHz PCM16 mono WAV on stop.
 * Does not send network requests. `options.signal` invalidates late getUserMedia.
 */
export async function startBahnarCapture(options = {}) {
  const getUserMedia = options.getUserMedia || defaultGetUserMedia
  const createContext = options.AudioContext
    ? (opts) => new options.AudioContext(opts)
    : defaultAudioContext
  const maxSeconds = options.maxSeconds || SPEECH_MAX_AUDIO_SECONDS
  const signal = options.signal

  const cancelled = () => Boolean(signal?.aborted || options.isCancelled?.())

  let stream
  try {
    stream = await getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    })
  } catch (err) {
    if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') {
      const denied = new Error('Cần cho phép micro để ghi tiếng Bahnar.')
      denied.code = 'PERMISSION_DENIED'
      throw denied
    }
    if (err?.code === 'UNSUPPORTED') throw err
    const fail = new Error(err?.message || 'Không bật được micro.')
    fail.code = 'CAPTURE_FAILED'
    throw fail
  }

  let ctx = null
  let processor = null
  let source = null
  let gain = null
  let limitTimer = null
  let finished = false
  let released = false
  const chunks = []
  const startedAt = Date.now()

  const cleanup = async () => {
    if (signal) {
      signal.removeEventListener('abort', onAbort)
    }
    if (released) return
    released = true
    if (limitTimer) {
      clearTimeout(limitTimer)
      limitTimer = null
    }
    try {
      processor?.disconnect()
    } catch {
      /* ignore */
    }
    try {
      source?.disconnect()
    } catch {
      /* ignore */
    }
    try {
      gain?.disconnect()
    } catch {
      /* ignore */
    }
    processor = null
    source = null
    gain = null
    stopTracks(stream)
    stream = null
    if (ctx && ctx.state !== 'closed' && typeof ctx.close === 'function') {
      try {
        await ctx.close()
      } catch {
        /* ignore */
      }
    }
    ctx = null
  }

  const cancelledError = () => {
    const err = new Error('Đã hủy ghi âm.')
    err.code = 'CAPTURE_CANCELLED'
    return err
  }

  const throwCancelled = async () => {
    await cleanup()
    throw cancelledError()
  }

  const onAbort = () => {
    finished = true
    stopTracks(stream)
    void cleanup()
  }
  if (signal) {
    if (signal.aborted) {
      stopTracks(stream)
      stream = null
      throw cancelledError()
    }
    signal.addEventListener('abort', onAbort, { once: true })
  }
  if (cancelled()) {
    stopTracks(stream)
    await throwCancelled()
  }

  try {
    ctx = createContext()
    if (cancelled()) await throwCancelled()
    if (typeof ctx.resume === 'function' && ctx.state === 'suspended') {
      try {
        await ctx.resume()
      } catch (err) {
        if (cancelled()) await throwCancelled()
        throw err
      }
      if (cancelled()) await throwCancelled()
    }
    source = ctx.createMediaStreamSource(stream)
    processor = ctx.createScriptProcessor(4096, 1, 1)
    gain = ctx.createGain()
    gain.gain.value = 0
    const maxInputSamples = Math.floor(maxSeconds * (ctx.sampleRate || 48000))
    let capturedSamples = 0
    processor.onaudioprocess = (event) => {
      if (finished || capturedSamples >= maxInputSamples) return
      const input = event.inputBuffer.getChannelData(0)
      const take = Math.min(input.length, maxInputSamples - capturedSamples)
      chunks.push(take === input.length ? Float32Array.from(input) : Float32Array.from(input.subarray(0, take)))
      capturedSamples += take
    }
    source.connect(processor)
    processor.connect(gain)
    gain.connect(ctx.destination)
    if (cancelled()) await throwCancelled()
  } catch (err) {
    await cleanup()
    if (err?.code === 'CAPTURE_CANCELLED') throw err
    wrapCaptureError(err, 'Không ghi được âm thanh.')
  }

  const session = {
    get elapsedMs() {
      return Date.now() - startedAt
    },
    get sampleRate() {
      return ctx?.sampleRate || 48000
    },
    async stopAndEncode() {
      if (finished) {
        const err = new Error('Bản ghi đã kết thúc.')
        err.code = 'ALREADY_STOPPED'
        throw err
      }
      finished = true
      if (processor) processor.onaudioprocess = null
      const rate = ctx?.sampleRate || 48000
      let encoded
      try {
        encoded = encodeCapturedSamples(chunks, rate)
      } finally {
        await cleanup()
      }
      return encoded
    },
    async cancel() {
      if (finished && released) return
      finished = true
      if (processor) processor.onaudioprocess = null
      chunks.length = 0
      await cleanup()
    },
  }

  limitTimer = setTimeout(() => {
    if (!finished && options.onMaxDuration) {
      options.onMaxDuration(session)
    }
  }, maxSeconds * 1000)

  if (cancelled()) {
    await session.cancel()
    throw cancelledError()
  }

  return session
}
