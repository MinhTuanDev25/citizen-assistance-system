import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { SPEECH_MAX_WAV_BYTES } from '../config/speech.js'
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
    'Gỡ file': 'Gỡ tệp',
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

describe('BahnarSpeechPanel', () => {
  let host

  beforeEach(() => {
    document.body.innerHTML = ''
  })

  afterEach(() => {
    host?.remove()
  })

  it('does not call translate until Dừng & Dịch, and cancel sends nothing', async () => {
    const translate = vi.fn()
    let capture
    const startCapture = vi.fn(async () => {
      capture = {
        elapsedMs: 0,
        stopAndEncode: vi.fn(async () => {
          const bytes = encodeWavPcm16Mono(tone(3200))
          return { bytes, durationMs: 200, sampleCount: 3200 }
        }),
        cancel: vi.fn(async () => {}),
      }
      return capture
    })
    host = await renderPanel({
      sessionId: 'sess-1',
      guestToken: 'g',
      startCapture,
      translate,
    })
    await act(async () => {
      buttonNamed(host, 'Micro Bahnar').click()
    })
    await act(async () => {
      buttonNamed(host, 'Bắt đầu ghi âm').click()
    })
    await flush()
    expect(startCapture).toHaveBeenCalledTimes(1)
    expect(translate).not.toHaveBeenCalled()
    await act(async () => {
      buttonNamed(host, 'Hủy ghi âm').click()
    })
    await flush()
    expect(capture.cancel).toHaveBeenCalledTimes(1)
    expect(translate).not.toHaveBeenCalled()
  })

  it('Stop & Translate sends one request then requires confirm before chat', async () => {
    const onConfirm = vi.fn(async () => true)
    const translate = vi.fn(async () => ({
      requestId: '11111111-1111-1111-1111-111111111111',
      asrTextBahnar: 'inh krĕ',
      translatedTextVi: 'tôi muốn khai sinh',
    }))
    const startCapture = vi.fn(async () => ({
      elapsedMs: 400,
      stopAndEncode: vi.fn(async () => ({
        bytes: encodeWavPcm16Mono(tone(3200)),
        durationMs: 200,
        sampleCount: 3200,
      })),
      cancel: vi.fn(async () => {}),
    }))
    host = await renderPanel({
      sessionId: 'sess-9',
      guestToken: 'tok',
      startCapture,
      translate,
      onConfirm,
    })
    await act(async () => {
      buttonNamed(host, 'Micro Bahnar').click()
    })
    await act(async () => {
      buttonNamed(host, 'Bắt đầu ghi âm').click()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Dừng & Dịch').click()
      buttonNamed(host, 'Dừng & Dịch').click()
    })
    await flush()
    expect(translate).toHaveBeenCalledTimes(1)
    expect(translate.mock.calls[0][0].sessionId).toBe('sess-9')
    expect(translate.mock.calls[0][0].guestToken).toBe('tok')
    expect(onConfirm).not.toHaveBeenCalled()
    const bahnar = host.querySelector('textarea[aria-label="Văn bản Bahnar"]')
    const viBox = host.querySelector('textarea[aria-label="Bản dịch tiếng Việt"]')
    expect(bahnar.value).toBe('inh krĕ')
    expect(viBox.value).toBe('tôi muốn khai sinh')
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
    expect(onConfirm.mock.calls[0][1]).toEqual({
      speechRequestId: '11111111-1111-1111-1111-111111111111',
    })
    await act(async () => {
      buttonNamed(host, 'Micro Bahnar').click()
    })
    await act(async () => {
      buttonNamed(host, 'Bắt đầu ghi âm').click()
    })
    await flush()
    expect(startCapture).toHaveBeenCalledTimes(2)
  })

  it('upload waits for Dịch âm thanh and rejects a non-WAV file', async () => {
    const translate = vi.fn()
    host = await renderPanel({
      sessionId: 'sess-1',
      guestToken: 'g',
      translate,
      startCapture: vi.fn(),
    })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    const input = host.querySelector('input[type="file"]')
    const bad = new File([new Uint8Array([73, 68, 51])], 'song.mp3', { type: 'audio/mpeg' })
    await act(async () => {
      Object.defineProperty(input, 'files', { configurable: true, value: [bad] })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await flush()
    expect(host.textContent).toMatch(/WAV/)
    expect(translate).not.toHaveBeenCalled()
    expect(buttonNamed(host, 'Dịch âm thanh').disabled).toBe(true)
  })

  it('valid WAV upload translates only after the button, cancel skips chat', async () => {
    const onConfirm = vi.fn()
    const translate = vi.fn(async () => ({
      asrTextBahnar: 'a',
      translatedTextVi: 'b',
    }))
    host = await renderPanel({
      sessionId: 'sess-3',
      translate,
      onConfirm,
      startCapture: vi.fn(),
    })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    const wavFile = new File([encodeWavPcm16Mono(tone(3200))], 'utt.wav', { type: 'audio/wav' })
    const input = host.querySelector('input[type="file"]')
    await act(async () => {
      Object.defineProperty(input, 'files', { configurable: true, value: [wavFile] })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await flush()
    expect(translate).not.toHaveBeenCalled()
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    expect(translate).toHaveBeenCalledTimes(1)
    await act(async () => {
      buttonNamed(host, 'Hủy').click()
    })
    expect(onConfirm).not.toHaveBeenCalled()
  })

  it('unmount releases the capture and sends no translate request', async () => {
    const translate = vi.fn()
    const cancel = vi.fn(async () => {})
    const startCapture = vi.fn(async () => ({
      elapsedMs: 0,
      stopAndEncode: vi.fn(),
      cancel,
    }))
    host = await renderPanel({
      sessionId: 'sess-u',
      startCapture,
      translate,
    })
    await act(async () => {
      buttonNamed(host, 'Micro Bahnar').click()
    })
    await act(async () => {
      buttonNamed(host, 'Bắt đầu ghi âm').click()
    })
    await flush()
    await act(async () => {
      host._root.unmount()
    })
    expect(cancel).toHaveBeenCalledTimes(1)
    expect(translate).not.toHaveBeenCalled()
  })

  it('stops a late permission grant after unmount, cancel, and mode switch', async () => {
    const translate = vi.fn()
    let grant
    const startCapture = vi.fn(
      () =>
        new Promise((resolve) => {
          grant = resolve
        }),
    )
    host = await renderPanel({ sessionId: 'sess-late', startCapture, translate })
    await act(async () => {
      buttonNamed(host, 'Micro Bahnar').click()
    })
    await act(async () => {
      buttonNamed(host, 'Bắt đầu ghi âm').click()
    })
    const stop = vi.fn()
    const cap = { elapsedMs: 0, stopAndEncode: vi.fn(), cancel: vi.fn(async () => { stop() }) }
    await act(async () => {
      host._root.unmount()
    })
    grant(cap)
    await flush()
    expect(cap.cancel).toHaveBeenCalled()
    expect(stop).toHaveBeenCalledTimes(1)
    expect(translate).not.toHaveBeenCalled()

    host = await renderPanel({ sessionId: 'sess-late2', startCapture, translate })
    await act(async () => {
      buttonNamed(host, 'Micro Bahnar').click()
    })
    await act(async () => {
      buttonNamed(host, 'Bắt đầu ghi âm').click()
    })
    const cap2 = { elapsedMs: 0, stopAndEncode: vi.fn(), cancel: vi.fn(async () => {}) }
    await act(async () => {
      buttonNamed(host, 'Hủy ghi âm').click()
    })
    grant(cap2)
    await flush()
    expect(cap2.cancel).toHaveBeenCalledTimes(1)

    await act(async () => {
      buttonNamed(host, 'Bắt đầu ghi âm').click()
    })
    const cap3 = { elapsedMs: 0, stopAndEncode: vi.fn(), cancel: vi.fn(async () => {}) }
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    grant(cap3)
    await flush()
    expect(cap3.cancel).toHaveBeenCalledTimes(1)
  })

  it('does not wipe a new Upload file after a slow microphone cancel', async () => {
    const translate = vi.fn()
    let finishCancel
    const startCapture = vi.fn(async () => ({
      elapsedMs: 400,
      stopAndEncode: vi.fn(),
      cancel: vi.fn(
        () =>
          new Promise((resolve) => {
            finishCancel = resolve
          }),
      ),
    }))
    host = await renderPanel({ sessionId: 'sess-cancel-race', startCapture, translate })
    await act(async () => {
      buttonNamed(host, 'Micro Bahnar').click()
    })
    await act(async () => {
      buttonNamed(host, 'Bắt đầu ghi âm').click()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Hủy ghi âm').click()
    })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    const wav = encodeWavPcm16Mono(tone(3200))
    const input = host.querySelector('input[type="file"]')
    await act(async () => {
      Object.defineProperty(input, 'files', {
        configurable: true,
        value: [{ name: 'keep.wav', size: wav.byteLength, arrayBuffer: async () => wav.buffer }],
      })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await flush()
    expect(host.textContent).toContain('keep.wav')
    await act(async () => {
      finishCancel()
    })
    await flush()
    expect(host.textContent).toContain('keep.wav')
    expect(host.textContent).not.toMatch(/Đã hủy ghi âm/)
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    expect(translate).toHaveBeenCalledTimes(1)
    expect(translate.mock.calls[0][0].filename).toBe('keep.wav')
  })

  it('ignores a second Start while permission is pending', async () => {
    let grant
    const startCapture = vi.fn(
      () =>
        new Promise((resolve) => {
          grant = resolve
        }),
    )
    host = await renderPanel({ sessionId: 'sess-dbl', startCapture, translate: vi.fn() })
    await act(async () => {
      buttonNamed(host, 'Micro Bahnar').click()
    })
    await act(async () => {
      buttonNamed(host, 'Bắt đầu ghi âm').click()
      buttonNamed(host, 'Bắt đầu ghi âm').click()
    })
    expect(startCapture).toHaveBeenCalledTimes(1)
    grant({ elapsedMs: 0, stopAndEncode: vi.fn(), cancel: vi.fn(async () => {}) })
    await flush()
  })

  it('keeps the latest selected file when an older read finishes later', async () => {
    const translate = vi.fn()
    host = await renderPanel({ sessionId: 'sess-race', startCapture: vi.fn(), translate })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    const wavA = encodeWavPcm16Mono(tone(3200))
    const wavB = encodeWavPcm16Mono(tone(4800))
    let finishA
    let finishB
    const fileA = {
      name: 'a.wav',
      size: wavA.byteLength,
      arrayBuffer: () => new Promise((resolve) => { finishA = () => resolve(wavA.buffer) }),
    }
    const fileB = {
      name: 'b.wav',
      size: wavB.byteLength,
      arrayBuffer: () => new Promise((resolve) => { finishB = () => resolve(wavB.buffer) }),
    }
    const input = host.querySelector('input[type="file"]')
    await act(async () => {
      Object.defineProperty(input, 'files', { configurable: true, value: [fileA] })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await act(async () => {
      Object.defineProperty(input, 'files', { configurable: true, value: [fileB] })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await act(async () => {
      finishB()
    })
    await flush()
    expect(host.textContent).toContain('b.wav')
    await act(async () => {
      finishA()
    })
    await flush()
    expect(host.textContent).toContain('b.wav')
    expect(host.textContent).not.toContain('a.wav')
  })

  it('does not submit audio A while replacement B is still reading', async () => {
    const translate = vi.fn()
    host = await renderPanel({ sessionId: 'sess-stale-audio', startCapture: vi.fn(), translate })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    const wavA = encodeWavPcm16Mono(tone(3200))
    const wavB = encodeWavPcm16Mono(tone(4800))
    const input = host.querySelector('input[type="file"]')
    await act(async () => {
      Object.defineProperty(input, 'files', {
        configurable: true,
        value: [{ name: 'a.wav', size: wavA.byteLength, arrayBuffer: async () => wavA.buffer }],
      })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await flush()
    let finishB
    await act(async () => {
      Object.defineProperty(input, 'files', {
        configurable: true,
        value: [{
          name: 'b.wav',
          size: wavB.byteLength,
          arrayBuffer: () => new Promise((resolve) => { finishB = () => resolve(wavB.buffer) }),
        }],
      })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    expect(translate).not.toHaveBeenCalled()
    await act(async () => {
      finishB()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    expect(translate).toHaveBeenCalledTimes(1)
    expect(translate.mock.calls[0][0].filename).toBe('b.wav')
    expect(translate.mock.calls[0][0].audio.byteLength).toBe(wavB.byteLength)
  })

  it('does not reuse A after an invalid replacement B', async () => {
    const translate = vi.fn()
    host = await renderPanel({ sessionId: 'sess-invalid-b', startCapture: vi.fn(), translate })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    const wavA = encodeWavPcm16Mono(tone(3200))
    const input = host.querySelector('input[type="file"]')
    await act(async () => {
      Object.defineProperty(input, 'files', {
        configurable: true,
        value: [{ name: 'a.wav', size: wavA.byteLength, arrayBuffer: async () => wavA.buffer }],
      })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await flush()
    const bad = new File([new Uint8Array([73, 68, 51])], 'b.mp3', { type: 'audio/mpeg' })
    await act(async () => {
      Object.defineProperty(input, 'files', { configurable: true, value: [bad] })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    expect(translate).not.toHaveBeenCalled()
  })

  it('does not apply a file read after cancel or mode switch', async () => {
    host = await renderPanel({ sessionId: 'sess-file-cancel', startCapture: vi.fn(), translate: vi.fn() })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    const wav = encodeWavPcm16Mono(tone(3200))
    let finish
    const file = {
      name: 'late.wav',
      size: wav.byteLength,
      arrayBuffer: () => new Promise((resolve) => { finish = () => resolve(wav.buffer) }),
    }
    const input = host.querySelector('input[type="file"]')
    await act(async () => {
      Object.defineProperty(input, 'files', { configurable: true, value: [file] })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await act(async () => {
      buttonNamed(host, 'Gỡ file').click()
    })
    await act(async () => {
      finish()
    })
    await flush()
    expect(host.textContent).not.toContain('late.wav')
  })

  it('rejects oversized and empty files before arrayBuffer', async () => {
    host = await renderPanel({ sessionId: 'sess-size', startCapture: vi.fn(), translate: vi.fn() })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    const arrayBuffer = vi.fn(async () => new ArrayBuffer(8))
    const huge = { name: 'huge.wav', size: SPEECH_MAX_WAV_BYTES + 1, arrayBuffer }
    const input = host.querySelector('input[type="file"]')
    await act(async () => {
      Object.defineProperty(input, 'files', { configurable: true, value: [huge] })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await flush()
    expect(arrayBuffer).not.toHaveBeenCalled()
    expect(host.textContent).toMatch(/dung lượng/)
    const emptyBuf = vi.fn(async () => new ArrayBuffer(0))
    await act(async () => {
      Object.defineProperty(input, 'files', { configurable: true, value: [{ name: 'e.wav', size: 0, arrayBuffer: emptyBuf }] })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await flush()
    expect(emptyBuf).not.toHaveBeenCalled()
    const wav = encodeWavPcm16Mono(tone(3200))
    await act(async () => {
      Object.defineProperty(input, 'files', {
        configurable: true,
        value: [{ name: 'ok.wav', size: wav.byteLength, arrayBuffer: async () => wav.buffer }],
      })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await flush()
    expect(host.textContent).toContain('ok.wav')
  })

  it('surfaces a file read failure without sending audio', async () => {
    const translate = vi.fn()
    host = await renderPanel({ sessionId: 'sess-read', startCapture: vi.fn(), translate })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    const wav = encodeWavPcm16Mono(tone(3200))
    const input = host.querySelector('input[type="file"]')
    await act(async () => {
      Object.defineProperty(input, 'files', {
        configurable: true,
        value: [{
          name: 'bad.wav',
          size: wav.byteLength,
          arrayBuffer: async () => {
            throw new Error('read fail')
          },
        }],
      })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await flush()
    expect(host.textContent).toMatch(/Không đọc được tệp/)
    expect(translate).not.toHaveBeenCalled()
  })

  it('cancels an in-flight translation and ignores a late response', async () => {
    const onConfirm = vi.fn()
    let finish
    const translate = vi.fn(
      () =>
        new Promise((resolve) => {
          finish = resolve
        }),
    )
    host = await renderPanel({ sessionId: 'sess-abort', startCapture: vi.fn(), translate, onConfirm })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    const wavFile = new File([encodeWavPcm16Mono(tone(3200))], 'utt.wav', { type: 'audio/wav' })
    const input = host.querySelector('input[type="file"]')
    await act(async () => {
      Object.defineProperty(input, 'files', { configurable: true, value: [wavFile] })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    expect(buttonNamed(host, 'Hủy yêu cầu dịch')).toBeTruthy()
    await act(async () => {
      buttonNamed(host, 'Hủy yêu cầu dịch').click()
      buttonNamed(host, 'Hủy yêu cầu dịch')?.click()
    })
    await flush()
    await act(async () => {
      finish({ asrTextBahnar: 'stale', translatedTextVi: 'cũ' })
    })
    await flush()
    expect(host.querySelector('textarea[aria-label="Văn bản Bahnar"]')).toBeNull()
    expect(onConfirm).not.toHaveBeenCalled()
    expect(host.textContent).toMatch(/hủy yêu cầu dịch/i)
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    expect(translate).toHaveBeenCalledTimes(2)
  })

  it('aborts translation on unmount and mode switch', async () => {
    let finish
    const translate = vi.fn(
      () =>
        new Promise((resolve) => {
          finish = resolve
        }),
    )
    host = await renderPanel({ sessionId: 'sess-unmount-tx', startCapture: vi.fn(), translate, onConfirm: vi.fn() })
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    const wavFile = new File([encodeWavPcm16Mono(tone(3200))], 'utt.wav', { type: 'audio/wav' })
    const input = host.querySelector('input[type="file"]')
    await act(async () => {
      Object.defineProperty(input, 'files', { configurable: true, value: [wavFile] })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Dịch âm thanh').click()
    })
    await flush()
    await act(async () => {
      buttonNamed(host, 'Micro Bahnar').click()
    })
    await act(async () => {
      finish({ asrTextBahnar: 'x', translatedTextVi: 'y' })
    })
    await flush()
    expect(host.querySelector('textarea[aria-label="Văn bản Bahnar"]')).toBeNull()
  })
})

