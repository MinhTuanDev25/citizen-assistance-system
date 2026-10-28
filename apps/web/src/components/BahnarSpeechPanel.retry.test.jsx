import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { BahnarSpeechPanel } from './BahnarSpeechPanel.jsx'
import { encodeWavPcm16Mono } from '../lib/wav.js'

function tone(n) {
  const samples = new Float32Array(n)
  samples.fill(0.1)
  return samples
}

function buttonNamed(host, label) {
  const translated = {
    'Micro Bahnar': 'Ghi tiếng Bahnar',
    'Tải file WAV': 'Tải tệp âm thanh',
    'Dừng & Dịch': 'Dừng và dịch',
  }[label] || label
  return [...host.querySelectorAll('button')].find((node) => node.textContent.replace(/\s+/g, ' ').trim() === translated)
}

async function renderPanel(props) {
  const host = document.createElement('div')
  document.body.appendChild(host)
  const root = createRoot(host)
  await act(async () => {
    root.render(createElement(BahnarSpeechPanel, props))
    await Promise.resolve()
  })
  host._root = root
  return host
}

async function flush() {
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()
  })
}

function okResult(overrides = {}) {
  return {
    asrTextBahnar: 'inh krĕ',
    translatedTextVi: 'tôi muốn khai sinh',
    ...overrides,
  }
}

function http500() {
  return Object.assign(new Error('Máy chủ gặp lỗi khi dịch giọng. Thử lại sau.'), {
    status: 500,
    code: 'SPEECH_UNAVAILABLE',
  })
}

function timeoutErr() {
  return Object.assign(new Error('Dịch giọng quá lâu. Thử bản ghi ngắn hơn.'), {
    code: 'SPEECH_TIMEOUT',
    name: 'AbortError',
  })
}

function networkErr() {
  return Object.assign(new Error('Mất kết nối tới máy chủ. Kiểm tra API đang chạy.'), {
    code: 'NETWORK',
  })
}

function validationErr() {
  return Object.assign(new Error('File âm thanh không hợp lệ. Chỉ nhận WAV PCM 16 kHz mono.'), {
    status: 415,
    code: 'VALIDATION',
  })
}

function unauthorizedErr() {
  return Object.assign(new Error('Thiếu hoặc sai thông tin đăng nhập phiên chat.'), {
    status: 401,
    code: 'UNAUTHORIZED',
  })
}

function micHarness() {
  const bytesA = encodeWavPcm16Mono(tone(3200))
  const bytesB = encodeWavPcm16Mono(tone(4800))
  const sessions = []
  const startCapture = vi.fn(async (opts) => {
    const index = sessions.length
    const bytes = index === 0 ? bytesA : bytesB
    const session = {
      elapsedMs: 30_000,
      stopAndEncode: vi.fn(async () => ({
        bytes,
        durationMs: index === 0 ? 200 : 300,
        sampleCount: index === 0 ? 3200 : 4800,
      })),
      cancel: vi.fn(async () => {}),
    }
    session.triggerMax = () => opts.onMaxDuration?.(session)
    sessions.push(session)
    return session
  })
  return { startCapture, sessions, bytesA, bytesB }
}

async function startMic(host) {
  await act(async () => {
    buttonNamed(host, 'Micro Bahnar').click()
  })
  await act(async () => {
    buttonNamed(host, 'Bắt đầu ghi âm').click()
  })
  await flush()
}

async function pickWav(host, bytes, name = 'utt.wav') {
  const input = host.querySelector('input[type="file"]')
  const wavFile = new File([bytes], name, { type: 'audio/wav' })
  await act(async () => {
    Object.defineProperty(input, 'files', { configurable: true, value: [wavFile] })
    input.dispatchEvent(new Event('change', { bubbles: true }))
  })
  await flush()
}

describe('BahnarSpeechPanel microphone generation and retry', () => {
  let host

  beforeEach(() => {
    document.body.innerHTML = ''
  })

  afterEach(() => {
    host?.remove()
  })

  it('H1 manual Stop & Translate sends exactly one request', async () => {
    const { startCapture, bytesA } = micHarness()
    const translate = vi.fn(async () => okResult())
    host = await renderPanel({ sessionId: 'sess-h1', guestToken: 'g', startCapture, translate })
    await startMic(host)
    await act(async () => {
      buttonNamed(host, 'Dừng & Dịch').click()
    })
    await flush()
    expect(translate).toHaveBeenCalledTimes(1)
    expect(translate.mock.calls[0][0].audio.byteLength).toBe(bytesA.byteLength)
    expect(translate.mock.calls[0][0].filename).toBe('microphone.wav')
    expect(host.querySelector('textarea[aria-label="Bản dịch tiếng Việt"]')).toBeTruthy()
  })

  it('H2 auto-stop at the limit then Translate sends the finalized WAV once', async () => {
    const { startCapture, sessions, bytesA } = micHarness()
    const translate = vi.fn(async () => okResult())
    host = await renderPanel({ sessionId: 'sess-h2', guestToken: 'g', startCapture, translate })
    await startMic(host)
    await act(async () => {
      sessions[0].triggerMax()
    })
    await flush()
    expect(translate).not.toHaveBeenCalled()
    expect(host.textContent).toMatch(/Đã ghi xong/)
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    expect(translate).toHaveBeenCalledTimes(1)
    expect(translate.mock.calls[0][0].audio).toBe(bytesA)
    expect(translate.mock.calls[0][0].filename).toBe('microphone.wav')
  })

  it('H3 auto-stop then rapid Translate clicks send one in-flight request', async () => {
    const { startCapture, sessions } = micHarness()
    let finish
    const translate = vi.fn(
      () =>
        new Promise((resolve) => {
          finish = resolve
        }),
    )
    host = await renderPanel({ sessionId: 'sess-h3', startCapture, translate })
    await startMic(host)
    await act(async () => {
      sessions[0].triggerMax()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    expect(translate).toHaveBeenCalledTimes(1)
    await act(async () => {
      finish(okResult())
    })
    await flush()
  })

  it('H4 finalized microphone audio is owned by the current generation', async () => {
    const { startCapture, sessions, bytesA } = micHarness()
    const translate = vi.fn(async () => okResult())
    host = await renderPanel({ sessionId: 'sess-h4', startCapture, translate })
    await startMic(host)
    await act(async () => {
      sessions[0].triggerMax()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    expect(translate).toHaveBeenCalledTimes(1)
    expect(translate.mock.calls[0][0].audio.byteLength).toBe(bytesA.byteLength)
  })

  it('H5 cancel microphone translation then retry the same WAV', async () => {
    const { startCapture, sessions, bytesA } = micHarness()
    let finish
    const translate = vi.fn(
      () =>
        new Promise((resolve) => {
          finish = resolve
        }),
    )
    host = await renderPanel({ sessionId: 'sess-h5', startCapture, translate })
    await startMic(host)
    await act(async () => {
      sessions[0].triggerMax()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Hủy yêu cầu dịch').click()
    })
    await flush()
    await act(async () => {
      finish(okResult({ asrTextBahnar: 'stale', translatedTextVi: 'cũ' }))
    })
    await flush()
    expect(host.querySelector('textarea[aria-label="Văn bản Bahnar"]')).toBeNull()
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    expect(translate).toHaveBeenCalledTimes(2)
    expect(translate.mock.calls[1][0].audio).toBe(bytesA)
  })

  it('H6 retranslate after success without submitting chat', async () => {
    const { startCapture, sessions } = micHarness()
    const onConfirm = vi.fn()
    const translate = vi.fn(async () => okResult())
    host = await renderPanel({ sessionId: 'sess-h6', startCapture, translate, onConfirm })
    await startMic(host)
    await act(async () => {
      sessions[0].triggerMax()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Dịch lại / Ghi âm lại').click()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    expect(translate).toHaveBeenCalledTimes(2)
    expect(onConfirm).not.toHaveBeenCalled()
  })

  it('H7 switching to Upload does not send stale microphone audio', async () => {
    const { startCapture, sessions } = micHarness()
    const translate = vi.fn(async () => okResult())
    host = await renderPanel({ sessionId: 'sess-h7', startCapture, translate })
    await startMic(host)
    await act(async () => {
      sessions[0].triggerMax()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh')?.click()
    })
    expect(translate).not.toHaveBeenCalled()
    expect(buttonNamed(host, 'Dịch âm thanh').disabled).toBe(true)
  })

  it('H8 a new recording replaces the previous capture', async () => {
    const { startCapture, sessions, bytesB } = micHarness()
    const translate = vi.fn(async () => okResult())
    host = await renderPanel({ sessionId: 'sess-h8', startCapture, translate })
    await startMic(host)
    await act(async () => {
      sessions[0].triggerMax()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Bắt đầu ghi âm').click()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Dừng & Dịch').click()
    })
    await flush()
    expect(translate).toHaveBeenCalledTimes(1)
    expect(translate.mock.calls[0][0].audio).toBe(bytesB)
  })

  it('H9 existing WAV upload still translates the latest file', async () => {
    const translate = vi.fn(async () => okResult())
    host = await renderPanel({ sessionId: 'sess-h9', startCapture: vi.fn(), translate })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    const wavA = encodeWavPcm16Mono(tone(3200))
    const wavB = encodeWavPcm16Mono(tone(4800))
    await pickWav(host, wavA, 'a.wav')
    await pickWav(host, wavB, 'b.wav')
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    expect(translate).toHaveBeenCalledTimes(1)
    expect(translate.mock.calls[0][0].filename).toBe('b.wav')
    expect(translate.mock.calls[0][0].audio.byteLength).toBe(wavB.byteLength)
  })

  it('M1 HTTP 500 shows Thử dịch lại and resubmits the same audio', async () => {
    const translate = vi.fn().mockRejectedValueOnce(http500()).mockResolvedValueOnce(okResult())
    const onConfirm = vi.fn()
    host = await renderPanel({ sessionId: 'sess-m1', startCapture: vi.fn(), translate, onConfirm })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    const wav = encodeWavPcm16Mono(tone(3200))
    await pickWav(host, wav, 'utt.wav')
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    expect(host.textContent).toMatch(/Máy chủ gặp lỗi/)
    expect(buttonNamed(host, 'Thử dịch lại')).toBeTruthy()
    await act(async () => {
      buttonNamed(host, 'Thử dịch lại').click()
    })
    await flush()
    expect(translate).toHaveBeenCalledTimes(2)
    expect(translate.mock.calls[1][0].audio.byteLength).toBe(wav.byteLength)
    expect(onConfirm).not.toHaveBeenCalled()
  })

  it('M2 timeout can be retried', async () => {
    const translate = vi.fn().mockRejectedValueOnce(timeoutErr()).mockResolvedValueOnce(okResult())
    host = await renderPanel({ sessionId: 'sess-m2', startCapture: vi.fn(), translate })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    await pickWav(host, encodeWavPcm16Mono(tone(3200)))
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Thử dịch lại').click()
    })
    await flush()
    expect(translate).toHaveBeenCalledTimes(2)
  })

  it('M3 network failure can be retried without recording again', async () => {
    const { startCapture, sessions, bytesA } = micHarness()
    const translate = vi.fn().mockRejectedValueOnce(networkErr()).mockResolvedValueOnce(okResult())
    host = await renderPanel({ sessionId: 'sess-m3', startCapture, translate })
    await startMic(host)
    await act(async () => {
      sessions[0].triggerMax()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    expect(startCapture).toHaveBeenCalledTimes(1)
    await act(async () => {
      buttonNamed(host, 'Thử dịch lại').click()
    })
    await flush()
    expect(startCapture).toHaveBeenCalledTimes(1)
    expect(translate.mock.calls[1][0].audio).toBe(bytesA)
  })

  it('M4 rapid Retry clicks keep a single in-flight request', async () => {
    let finish
    const translate = vi.fn().mockImplementationOnce(async () => {
      throw http500()
    }).mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve
        }),
    )
    host = await renderPanel({ sessionId: 'sess-m4', startCapture: vi.fn(), translate })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    await pickWav(host, encodeWavPcm16Mono(tone(3200)))
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Thử dịch lại').click()
      buttonNamed(host, 'Thử dịch lại')?.click()
      buttonNamed(host, 'Thử dịch lại')?.click()
    })
    await flush()
    expect(translate).toHaveBeenCalledTimes(2)
    await act(async () => {
      finish(okResult())
    })
    await flush()
  })

  it('M5 cancel then retry the same valid upload', async () => {
    let finish
    const translate = vi.fn(
      () =>
        new Promise((resolve) => {
          finish = resolve
        }),
    )
    host = await renderPanel({ sessionId: 'sess-m5', startCapture: vi.fn(), translate })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    const wav = encodeWavPcm16Mono(tone(3200))
    await pickWav(host, wav)
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Hủy yêu cầu dịch').click()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    expect(translate).toHaveBeenCalledTimes(2)
    await act(async () => {
      finish(okResult())
    })
    await flush()
  })

  it('M6 a late response from a canceled request cannot overwrite retry B', async () => {
    let finishA
    let finishB
    let calls = 0
    const translate = vi.fn(
      () =>
        new Promise((resolve) => {
          calls += 1
          if (calls === 1) finishA = resolve
          else finishB = resolve
        }),
    )
    host = await renderPanel({ sessionId: 'sess-m6', startCapture: vi.fn(), translate })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    await pickWav(host, encodeWavPcm16Mono(tone(3200)))
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Hủy yêu cầu dịch').click()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    await act(async () => {
      finishA(okResult({ asrTextBahnar: 'A', translatedTextVi: 'cũ A' }))
    })
    await flush()
    expect(host.querySelector('textarea[aria-label="Văn bản Bahnar"]')).toBeNull()
    await act(async () => {
      finishB(okResult({ asrTextBahnar: 'B', translatedTextVi: 'mới B' }))
    })
    await flush()
    expect(host.querySelector('textarea[aria-label="Văn bản Bahnar"]').value).toBe('B')
    expect(host.querySelector('textarea[aria-label="Bản dịch tiếng Việt"]').value).toBe('mới B')
  })

  it('M7 switching mode after failure does not submit stale retry audio', async () => {
    const translate = vi.fn().mockRejectedValueOnce(http500())
    host = await renderPanel({ sessionId: 'sess-m7', startCapture: vi.fn(), translate })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    await pickWav(host, encodeWavPcm16Mono(tone(3200)))
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Micro Bahnar').click()
    })
    expect(buttonNamed(host, 'Thử dịch lại')).toBeUndefined()
    await act(async () => {
      buttonNamed(host, 'Thử dịch lại')?.click()
    })
    expect(translate).toHaveBeenCalledTimes(1)
  })

  it('M8 successful retry shows Bahnar and Vietnamese without a Chat Turn', async () => {
    const onConfirm = vi.fn()
    const translate = vi.fn().mockRejectedValueOnce(http500()).mockResolvedValueOnce(okResult())
    host = await renderPanel({ sessionId: 'sess-m8', startCapture: vi.fn(), translate, onConfirm })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    await pickWav(host, encodeWavPcm16Mono(tone(3200)))
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Thử dịch lại').click()
    })
    await flush()
    expect(host.querySelector('textarea[aria-label="Văn bản Bahnar"]').value).toBe('inh krĕ')
    expect(host.querySelector('textarea[aria-label="Bản dịch tiếng Việt"]').value).toBe('tôi muốn khai sinh')
    expect(onConfirm).not.toHaveBeenCalled()
  })

  it('M9 confirm after retry sends exactly one Chat Turn', async () => {
    const onConfirm = vi.fn(async () => true)
    const translate = vi.fn().mockRejectedValueOnce(http500()).mockResolvedValueOnce(okResult())
    host = await renderPanel({ sessionId: 'sess-m9', startCapture: vi.fn(), translate, onConfirm })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    await pickWav(host, encodeWavPcm16Mono(tone(3200)))
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Thử dịch lại').click()
    })
    await flush()
    const viBox = host.querySelector('textarea[aria-label="Bản dịch tiếng Việt"]')
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set
      setter.call(viBox, 'tôi muốn đăng ký khai sinh')
      viBox.dispatchEvent(new Event('input', { bubbles: true }))
    })
    await act(async () => {
      buttonNamed(host, 'Xác nhận & Hỏi trợ lý').click()
    })
    await flush()
    expect(onConfirm).toHaveBeenCalledTimes(1)
    expect(onConfirm.mock.calls[0][0]).toBe('tôi muốn đăng ký khai sinh')
  })

  it('does not offer Retry for an invalid local WAV', async () => {
    host = await renderPanel({ sessionId: 'sess-bad', startCapture: vi.fn(), translate: vi.fn() })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    const bad = new File([new Uint8Array([73, 68, 51])], 'song.mp3', { type: 'audio/mpeg' })
    const input = host.querySelector('input[type="file"]')
    await act(async () => {
      Object.defineProperty(input, 'files', { configurable: true, value: [bad] })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await flush()
    expect(buttonNamed(host, 'Thử dịch lại')).toBeUndefined()
  })

  it('does not offer Retry when the server rejects the WAV format', async () => {
    const translate = vi.fn().mockRejectedValueOnce(validationErr())
    host = await renderPanel({ sessionId: 'sess-415', startCapture: vi.fn(), translate })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    await pickWav(host, encodeWavPcm16Mono(tone(3200)))
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    expect(host.textContent).toMatch(/không hợp lệ/i)
    expect(buttonNamed(host, 'Thử dịch lại')).toBeUndefined()
  })

  it('does not offer Retry after SPEECH_NO_SPEECH; user can record again', async () => {
    const noSpeech = Object.assign(
      new Error('Không nhận ra tiếng nói. Hãy ghi âm lại, nói gần micro hơn, rồi bấm Dịch âm thanh.'),
      { status: 422, code: 'SPEECH_NO_SPEECH' },
    )
    const translate = vi.fn().mockRejectedValueOnce(noSpeech)
    host = await renderPanel({ sessionId: 'sess-silence', startCapture: vi.fn(), translate })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    await pickWav(host, encodeWavPcm16Mono(new Float32Array(3200)))
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    expect(host.textContent).toMatch(/ghi âm lại/)
    expect(buttonNamed(host, 'Thử dịch lại')).toBeUndefined()
    expect(host.querySelector('textarea[aria-label="Bản dịch tiếng Việt"]')).toBeNull()
    expect(host.querySelector('textarea[aria-label="Văn bản Bahnar"]')).toBeNull()
    expect(buttonNamed(host, 'Xác nhận & Hỏi trợ lý')).toBeUndefined()
    expect(buttonNamed(host, 'Chọn file WAV') || host.querySelector('input[type="file"]')).toBeTruthy()
  })

  it('does not offer Retry after HTTP 401', async () => {
    const translate = vi.fn().mockRejectedValueOnce(unauthorizedErr())
    host = await renderPanel({ sessionId: 'sess-401', startCapture: vi.fn(), translate })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    await pickWav(host, encodeWavPcm16Mono(tone(3200)))
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    expect(host.textContent).toMatch(/đăng nhập phiên chat/)
    expect(buttonNamed(host, 'Thử dịch lại')).toBeUndefined()
  })
})
