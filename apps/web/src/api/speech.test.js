import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { SPEECH_MODEL_ID } from '../config/speech.js'

vi.mock('../auth/auth.js', () => ({
  loadAuth: vi.fn(() => null),
}))

import { loadAuth } from '../auth/auth.js'
import {
  isRetryableSpeechError,
  parseSpeechResult,
  resetSpeechInflight,
  speechErrorMessage,
  translateSpeech,
} from './speech.js'

function okPayload(overrides = {}) {
  return {
    request_id: '11111111-1111-1111-1111-111111111111',
    data: {
      request_id: '11111111-1111-1111-1111-111111111111',
      model_id: SPEECH_MODEL_ID,
      asr_text_bahnar: 'inh',
      translated_text_vi: 'tôi',
      duration_ms: 14432,
      latency_ms: 1254,
      ...overrides,
    },
  }
}

describe('parseSpeechResult', () => {
  it('accepts the verified Go envelope', () => {
    const out = parseSpeechResult(okPayload())
    expect(out.asrTextBahnar).toBe('inh')
    expect(out.translatedTextVi).toBe('tôi')
    expect(out.modelId).toBe('c0_cascaded')
  })

  it('rejects missing fields and wrong model', () => {
    expect(() => parseSpeechResult({ data: { model_id: 'c0_cascaded' } })).toThrow()
    expect(() =>
      parseSpeechResult(okPayload({ model_id: 'other', asr_text_bahnar: 'a', translated_text_vi: 'b' })),
    ).toThrow()
    expect(() => parseSpeechResult('{')).toThrow()
  })
})

describe('translateSpeech', () => {
  beforeEach(() => {
    resetSpeechInflight()
    loadAuth.mockReturnValue(null)
  })

  afterEach(() => {
    resetSpeechInflight()
  })

  it('posts multipart audio with guest token and session path', async () => {
    const fetchImpl = vi.fn(async (url, opts) => {
      expect(url).toBe('/api/v1/sessions/sess-1/speech/translate')
      expect(opts.method).toBe('POST')
      expect(opts.headers['X-Guest-Token']).toBe('guest-a')
      expect(opts.headers['X-Request-ID']).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
      )
      expect(opts.headers['Content-Type']).toBeUndefined()
      expect(opts.body).toBeInstanceOf(FormData)
      expect(opts.body.get('audio')).toBeTruthy()
      return {
        ok: true,
        json: async () => okPayload(),
      }
    })
    const wav = new Uint8Array([82, 73, 70, 70, 1, 2, 3, 4])
    const out = await translateSpeech({
      sessionId: 'sess-1',
      guestToken: 'guest-a',
      audio: wav,
      filename: 'utt.wav',
      fetchImpl,
    })
    expect(out.translatedTextVi).toBe('tôi')
    expect(fetchImpl).toHaveBeenCalledTimes(1)
  })

  it('sends Bearer for logged-in users', async () => {
    loadAuth.mockReturnValue({ accessToken: 'jwt-token' })
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => okPayload(),
    }))
    await translateSpeech({
      sessionId: 'sess-2',
      audio: new Uint8Array([1, 2, 3]),
      fetchImpl,
    })
    expect(fetchImpl.mock.calls[0][1].headers.Authorization).toBe('Bearer jwt-token')
  })

  it('maps HTTP errors and malformed success', async () => {
    for (const [status, code] of [
      [401, 'UNAUTHORIZED'],
      [403, 'FORBIDDEN'],
      [404, 'NOT_FOUND'],
      [413, 'PAYLOAD_TOO_LARGE'],
      [415, 'VALIDATION'],
      [422, 'SPEECH_NO_SPEECH'],
      [429, 'SPEECH_OVERLOADED'],
      [503, 'SPEECH_UNAVAILABLE'],
    ]) {
      resetSpeechInflight()
      const fetchImpl = vi.fn(async () => ({
        ok: false,
        status,
        json: async () => ({ error: { code, message: 'x' } }),
      }))
      await expect(
        translateSpeech({ sessionId: 's', audio: new Uint8Array([1]), fetchImpl }),
      ).rejects.toMatchObject({ status, code })
    }
    resetSpeechInflight()
    const bad = vi.fn(async () => ({
      ok: true,
      json: async () => ({ data: { model_id: 'c0_cascaded', asr_text_bahnar: '', translated_text_vi: 'x' } }),
    }))
    await expect(
      translateSpeech({ sessionId: 's', audio: new Uint8Array([1]), fetchImpl: bad }),
    ).rejects.toMatchObject({ code: 'SPEECH_INVALID_UPSTREAM' })
  })

  it('prevents overlapping requests', async () => {
    let resolveFirst
    const fetchImpl = vi.fn(
      () =>
        new Promise((resolve) => {
          resolveFirst = resolve
        }),
    )
    const first = translateSpeech({
      sessionId: 's',
      audio: new Uint8Array([1]),
      fetchImpl,
    })
    await expect(
      translateSpeech({ sessionId: 's', audio: new Uint8Array([1]), fetchImpl }),
    ).rejects.toMatchObject({ code: 'SPEECH_OVERLOADED' })
    resolveFirst({ ok: true, json: async () => okPayload() })
    await first
  })

  it('turns abort into a timeout error without a fake transcript', async () => {
    const fetchImpl = vi.fn(async (_url, opts) => {
      const err = new Error('aborted')
      err.name = 'AbortError'
      opts.signal.addEventListener('abort', () => {})
      throw err
    })
    await expect(
      translateSpeech({
        sessionId: 's',
        audio: new Uint8Array([1]),
        fetchImpl,
        timeoutMs: 1,
      }),
    ).rejects.toMatchObject({ code: 'SPEECH_TIMEOUT' })
  })

  it('maps a user abort to SPEECH_CANCELLED', async () => {
    const user = new AbortController()
    user.abort()
    await expect(
      translateSpeech({
        sessionId: 's',
        audio: new Uint8Array([1]),
        signal: user.signal,
        fetchImpl: vi.fn(),
      }),
    ).rejects.toMatchObject({ code: 'SPEECH_CANCELLED' })
  })

  it('maps a thrown network error without inventing a transcript', async () => {
    const fetchImpl = vi.fn(async () => {
      throw new TypeError('Failed to fetch')
    })
    await expect(
      translateSpeech({ sessionId: 's', audio: new Uint8Array([1]), fetchImpl }),
    ).rejects.toMatchObject({ code: 'NETWORK' })
  })
})

describe('speechErrorMessage', () => {
  it('uses Vietnamese copy for known codes', () => {
    expect(speechErrorMessage({ code: 'SPEECH_DISABLED' })).toMatch(/tắt/)
    expect(speechErrorMessage({ status: 413 })).toMatch(/dung lượng/)
    expect(speechErrorMessage({ code: 'SPEECH_NO_SPEECH', status: 422 })).toMatch(/ghi âm lại/)
    expect(speechErrorMessage({ status: 502, code: 'SPEECH_UNAVAILABLE' })).toMatch(/RunPod|máy dịch/)
  })
})

describe('isRetryableSpeechError', () => {
  it('allows retry after transient inference and network failures', () => {
    expect(isRetryableSpeechError({ status: 500 })).toBe(true)
    expect(isRetryableSpeechError({ code: 'SPEECH_TIMEOUT' })).toBe(true)
    expect(isRetryableSpeechError({ code: 'NETWORK' })).toBe(true)
    expect(isRetryableSpeechError({ code: 'SPEECH_UNAVAILABLE' })).toBe(true)
  })

  it('does not retry invalid or oversized audio', () => {
    expect(isRetryableSpeechError({ code: 'VALIDATION' })).toBe(false)
    expect(isRetryableSpeechError({ code: 'PAYLOAD_TOO_LARGE' })).toBe(false)
    expect(isRetryableSpeechError({ status: 413 })).toBe(false)
    expect(isRetryableSpeechError({ status: 415 })).toBe(false)
  })

  it('does not retry silence / no-speech; the same WAV will fail again', () => {
    expect(isRetryableSpeechError({ code: 'SPEECH_NO_SPEECH', status: 422 })).toBe(false)
    expect(isRetryableSpeechError({ status: 422 })).toBe(false)
  })

  it('does not retry auth or missing-session errors', () => {
    expect(isRetryableSpeechError({ status: 401, code: 'UNAUTHORIZED' })).toBe(false)
    expect(isRetryableSpeechError({ status: 403, code: 'FORBIDDEN' })).toBe(false)
    expect(isRetryableSpeechError({ status: 404, code: 'NOT_FOUND' })).toBe(false)
    expect(isRetryableSpeechError({ status: 409, code: 'CONFLICT' })).toBe(false)
  })
})
