import { afterEach, describe, expect, it, vi } from 'vitest'
import { SPEECH_MAX_AUDIO_SECONDS, SPEECH_SAMPLE_RATE } from '../config/speech.js'
import { startBahnarCapture } from './micRecorder.js'
import { validateBahnarWav } from './wav.js'

class FakeAudioContext {
  constructor() {
    this.sampleRate = 48000
    this.state = 'running'
    this.processor = {
      onaudioprocess: null,
      connect: vi.fn(),
      disconnect: vi.fn(),
    }
    this.gain = { gain: { value: 1 }, connect: vi.fn(), disconnect: vi.fn() }
    this.source = { connect: vi.fn(), disconnect: vi.fn() }
  }

  createMediaStreamSource() {
    return this.source
  }

  createScriptProcessor() {
    return this.processor
  }

  createGain() {
    return this.gain
  }

  async close() {
    this.state = 'closed'
  }
}

function toneChunk(n = 4800) {
  return {
    inputBuffer: {
      getChannelData: () => {
        const data = new Float32Array(n)
        data.fill(0.2)
        return data
      },
    },
  }
}

describe('startBahnarCapture', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('captures then encodes once and stops tracks', async () => {
    const stop = vi.fn()
    const stream = { getTracks: () => [{ stop }] }
    const getUserMedia = vi.fn(async () => stream)
    const ctx = new FakeAudioContext()
    const session = await startBahnarCapture({
      getUserMedia,
      AudioContext: function AudioContext() {
        return ctx
      },
    })
    ctx.processor.onaudioprocess(toneChunk(48000))
    const encoded = await session.stopAndEncode()
    expect(encoded.bytes[0]).toBe(82)
    expect(encoded.durationMs).toBeGreaterThan(0)
    expect(stop).toHaveBeenCalledTimes(1)
    expect(ctx.state).toBe('closed')
    await expect(session.stopAndEncode()).rejects.toThrow(/kết thúc/)
  })

  it('cancel sends no encode and releases the microphone', async () => {
    const stop = vi.fn()
    const stream = { getTracks: () => [{ stop }] }
    const ctx = new FakeAudioContext()
    const session = await startBahnarCapture({
      getUserMedia: async () => stream,
      AudioContext: function AudioContext() {
        return ctx
      },
    })
    ctx.processor.onaudioprocess(toneChunk())
    await session.cancel()
    expect(stop).toHaveBeenCalledTimes(1)
    expect(ctx.state).toBe('closed')
    await session.cancel()
    expect(stop).toHaveBeenCalledTimes(1)
    await expect(session.stopAndEncode()).rejects.toThrow(/kết thúc/)
  })

  it('maps permission denial', async () => {
    const denied = Object.assign(new Error('no'), { name: 'NotAllowedError' })
    await expect(
      startBahnarCapture({
        getUserMedia: async () => {
          throw denied
        },
      }),
    ).rejects.toMatchObject({ code: 'PERMISSION_DENIED' })
  })

  it('fails closed when getUserMedia is missing', async () => {
    const original = navigator.mediaDevices
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: undefined })
    try {
      await expect(startBahnarCapture()).rejects.toMatchObject({ code: 'UNSUPPORTED' })
    } finally {
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: original })
    }
  })

  it('stops the acquired track when AudioContext construction fails', async () => {
    const stop = vi.fn()
    const stream = { getTracks: () => [{ stop }] }
    await expect(
      startBahnarCapture({
        getUserMedia: async () => stream,
        AudioContext: function AudioContext() {
          throw new Error('AudioContext blocked')
        },
      }),
    ).rejects.toThrow(/AudioContext blocked/)
    expect(stop).toHaveBeenCalledTimes(1)
  })

  it('stops tracks when AudioContext.resume fails', async () => {
    const stop = vi.fn()
    const stream = { getTracks: () => [{ stop }] }
    const ctx = new FakeAudioContext()
    ctx.state = 'suspended'
    ctx.resume = async () => {
      throw new Error('resume denied')
    }
    await expect(
      startBahnarCapture({
        getUserMedia: async () => stream,
        AudioContext: function AudioContext() {
          return ctx
        },
      }),
    ).rejects.toThrow(/resume denied/)
    expect(stop).toHaveBeenCalledTimes(1)
  })

  it('stops tracks when node creation or connect fails', async () => {
    const stop = vi.fn()
    const stream = { getTracks: () => [{ stop }] }
    const ctx = new FakeAudioContext()
    ctx.createMediaStreamSource = () => {
      throw new Error('source failed')
    }
    await expect(
      startBahnarCapture({
        getUserMedia: async () => stream,
        AudioContext: function AudioContext() {
          return ctx
        },
      }),
    ).rejects.toThrow(/source failed/)
    expect(stop).toHaveBeenCalledTimes(1)

    const stop2 = vi.fn()
    const ctx2 = new FakeAudioContext()
    ctx2.source.connect = () => {
      throw new Error('connect failed')
    }
    await expect(
      startBahnarCapture({
        getUserMedia: async () => ({ getTracks: () => [{ stop: stop2 }] }),
        AudioContext: function AudioContext() {
          return ctx2
        },
      }),
    ).rejects.toThrow(/connect failed/)
    expect(stop2).toHaveBeenCalledTimes(1)
  })

  it('stops tracks immediately while AudioContext.resume is still pending', async () => {
    const stop = vi.fn()
    const ac = new AbortController()
    let finishResume
    const ctx = new FakeAudioContext()
    ctx.state = 'suspended'
    ctx.resume = vi.fn(
      () =>
        new Promise((resolve) => {
          finishResume = resolve
        }),
    )
    const pending = startBahnarCapture({
      signal: ac.signal,
      getUserMedia: async () => ({ getTracks: () => [{ stop }] }),
      AudioContext: function AudioContext() {
        return ctx
      },
    })
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()
    expect(ctx.resume).toHaveBeenCalled()
    expect(finishResume).toEqual(expect.any(Function))
    expect(stop).not.toHaveBeenCalled()
    ac.abort()
    ac.abort()
    expect(stop.mock.calls.length).toBeGreaterThanOrEqual(1)
    expect(finishResume).toEqual(expect.any(Function))
    finishResume()
    await expect(pending).rejects.toMatchObject({ code: 'CAPTURE_CANCELLED' })
  })

  it('encodes a 30s WAV when extra ScriptProcessor buffers arrive before stop', async () => {
    const stop = vi.fn()
    const ctx = new FakeAudioContext()
    const session = await startBahnarCapture({
      maxSeconds: SPEECH_MAX_AUDIO_SECONDS,
      getUserMedia: async () => ({ getTracks: () => [{ stop }] }),
      AudioContext: function AudioContext() {
        return ctx
      },
    })
    const block = 4096
    const blocks = Math.ceil((30.2 * ctx.sampleRate) / block)
    for (let i = 0; i < blocks; i += 1) {
      ctx.processor.onaudioprocess(toneChunk(block))
    }
    const encoded = await session.stopAndEncode()
    expect(encoded.sampleCount).toBe(SPEECH_SAMPLE_RATE * SPEECH_MAX_AUDIO_SECONDS)
    expect(encoded.durationMs).toBe(SPEECH_MAX_AUDIO_SECONDS * 1000)
    expect(validateBahnarWav(encoded.bytes).ok).toBe(true)
    expect(stop).toHaveBeenCalled()
  })

  it('still encodes after a late max-duration timer', async () => {
    vi.useFakeTimers()
    const ctx = new FakeAudioContext()
    let autoSession
    const pending = startBahnarCapture({
      maxSeconds: SPEECH_MAX_AUDIO_SECONDS,
      getUserMedia: async () => ({ getTracks: () => [{ stop: vi.fn() }] }),
      AudioContext: function AudioContext() {
        return ctx
      },
      onMaxDuration: (session) => {
        autoSession = session
      },
    })
    const session = await pending
    const block = 4096
    const blocks = Math.ceil((30.2 * ctx.sampleRate) / block)
    for (let i = 0; i < blocks; i += 1) {
      ctx.processor.onaudioprocess(toneChunk(block))
    }
    await vi.advanceTimersByTimeAsync(30_100)
    expect(autoSession).toBe(session)
    const encoded = await session.stopAndEncode()
    expect(encoded.durationMs).toBe(30_000)
    expect(encoded.sampleCount).toBe(480_000)
  })

  it('stops tracks if permission is granted after abort', async () => {
    const stop = vi.fn()
    const ac = new AbortController()
    let grant
    const pending = startBahnarCapture({
      signal: ac.signal,
      getUserMedia: () =>
        new Promise((resolve) => {
          grant = resolve
        }),
      AudioContext: FakeAudioContext,
    })
    ac.abort()
    grant({ getTracks: () => [{ stop }] })
    await expect(pending).rejects.toMatchObject({ code: 'CAPTURE_CANCELLED' })
    expect(stop).toHaveBeenCalledTimes(1)
  })
})
