import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { encodeWavPcm16Mono } from '../lib/wav.js'

const session = vi.hoisted(() => ({
  ensureSession: vi.fn(),
  executeChatTurn: vi.fn(),
  listSessionMessages: vi.fn(),
  startFreshSession: vi.fn(),
}))

const speech = vi.hoisted(() => ({
  translateSpeech: vi.fn(),
}))

const mic = vi.hoisted(() => ({
  startBahnarCapture: vi.fn(),
}))

vi.mock('../api/session.js', async (importOriginal) => {
  const actual = await importOriginal()
  return {
    ...actual,
    ensureSession: session.ensureSession,
    executeChatTurn: session.executeChatTurn,
    listSessionMessages: session.listSessionMessages,
    startFreshSession: session.startFreshSession,
  }
})
vi.mock('../api/speech.js', () => ({
  translateSpeech: speech.translateSpeech,
}))
vi.mock('../lib/micRecorder.js', () => ({
  startBahnarCapture: mic.startBahnarCapture,
}))
vi.mock('../api/catalog.js', () => ({
  listProcedures: vi.fn(async () => ({ domains: [] })),
}))
vi.mock('../auth/AuthContext.jsx', () => ({
  useAuth: () => ({ user: null, logout: () => {} }),
}))
vi.mock('../commune/CommuneContext.jsx', () => ({
  useCommune: () => ({ xaId: 'xa_chu_se', commune: { name: 'Chư Sê' } }),
}))
vi.mock('../components/CitizenLayout.jsx', () => ({
  CitizenLayout: ({ children }) => children,
}))

function buttonNamed(host, label) {
  const translated = {
    'Micro Bahnar': 'Ghi tiếng Bahnar',
    'Tải file WAV': 'Tải tệp âm thanh',
    'Dừng & Dịch': 'Dừng và dịch',
  }[label] || label
  return [...host.querySelectorAll('button')].find(
    (node) => node.textContent.replace(/\s+/g, ' ').trim() === translated,
  )
}

function setInputValue(node, value) {
  const proto = node.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype
  Object.getOwnPropertyDescriptor(proto, 'value').set.call(node, value)
  node.dispatchEvent(new Event('input', { bubbles: true }))
}

async function flush() {
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()
  })
}

async function renderPage() {
  const { default: Page } = await import('./CitizenChatPage.jsx')
  const host = document.createElement('div')
  document.body.appendChild(host)
  await act(async () => {
    createRoot(host).render(createElement(MemoryRouter, { initialEntries: ['/chat'] }, createElement(Page)))
    await Promise.resolve()
  })
  await flush()
  return host
}

describe('CitizenChatPage speech confirmation', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
    session.ensureSession.mockReset()
    session.executeChatTurn.mockReset()
    session.listSessionMessages.mockReset()
    session.startFreshSession.mockReset()
    speech.translateSpeech.mockReset()
    mic.startBahnarCapture.mockReset()
    session.ensureSession.mockResolvedValue({
      sessionId: 'sess-1',
      guestToken: 'g-1',
      xaId: 'xa_chu_se',
    })
    session.listSessionMessages.mockResolvedValue({ items: [] })
    session.executeChatTurn.mockResolvedValue({ ok: true, history: { items: [] } })
    speech.translateSpeech.mockResolvedValue({
      requestId: '11111111-1111-1111-1111-111111111111',
      asrTextBahnar: 'inh krĕ',
      translatedTextVi: 'tôi muốn khai sinh',
    })
    mic.startBahnarCapture.mockResolvedValue({
      elapsedMs: 300,
      stopAndEncode: vi.fn(async () => ({
        bytes: encodeWavPcm16Mono(new Float32Array(3200).fill(0.1)),
        durationMs: 200,
        sampleCount: 3200,
      })),
      cancel: vi.fn(async () => {}),
    })
  })

  it('keeps Vietnamese browser dictation and typed chat on the existing session', async () => {
    const host = await renderPage()
    const viMic = host.querySelector('button[aria-label="Nói câu hỏi bằng tiếng Việt"]')
    expect(viMic).toBeTruthy()
    expect(host.textContent).toContain('Ghi tiếng Bahnar')
    const composer = host.querySelector('textarea[aria-label="Nội dung câu hỏi"]')
    await act(async () => {
      setInputValue(composer, 'xin chào cán bộ')
    })
    await act(async () => {
      buttonNamed(host, 'Gửi').click()
    })
    await flush()
    expect(session.executeChatTurn).toHaveBeenCalledTimes(1)
    expect(session.executeChatTurn.mock.calls[0][0]).toMatchObject({
      sessionId: 'sess-1',
      guestToken: 'g-1',
      message: 'xin chào cán bộ',
    })
    expect(session.executeChatTurn.mock.calls[0][0].speechRequestId).toBeUndefined()
    expect(speech.translateSpeech).not.toHaveBeenCalled()
  })

  it('does not post a Chat Turn until the Vietnamese translation is confirmed', async () => {
    const host = await renderPage()
    await act(async () => {
      buttonNamed(host, 'Micro Bahnar').click()
    })
    await act(async () => {
      buttonNamed(host, 'Bắt đầu ghi âm').click()
    })
    await flush()
    expect(speech.translateSpeech).not.toHaveBeenCalled()
    await act(async () => {
      buttonNamed(host, 'Dừng & Dịch').click()
    })
    await flush()
    expect(speech.translateSpeech).toHaveBeenCalledTimes(1)
    expect(speech.translateSpeech.mock.calls[0][0]).toMatchObject({
      sessionId: 'sess-1',
      guestToken: 'g-1',
    })
    expect(session.executeChatTurn).not.toHaveBeenCalled()
    const viBox = host.querySelector('textarea[aria-label="Bản dịch tiếng Việt"]')
    await act(async () => {
      setInputValue(viBox, 'tôi muốn đăng ký khai sinh')
    })
    await act(async () => {
      buttonNamed(host, 'Xác nhận & Hỏi trợ lý').click()
    })
    await flush()
    expect(session.executeChatTurn).toHaveBeenCalledTimes(1)
    expect(session.executeChatTurn.mock.calls[0][0].message).toBe('tôi muốn đăng ký khai sinh')
    expect(session.executeChatTurn.mock.calls[0][0].speechRequestId).toBe(
      '11111111-1111-1111-1111-111111111111',
    )
    expect(session.ensureSession).toHaveBeenCalledTimes(1)
  })

  it('failed Chat Turn after confirm does not leave a false success', async () => {
    session.executeChatTurn.mockResolvedValue({
      ok: false,
      error: new Error('Không gửi được tin nhắn'),
    })
    const host = await renderPage()
    await act(async () => {
      buttonNamed(host, 'Tải file WAV').click()
    })
    const wavFile = new File(
      [encodeWavPcm16Mono(new Float32Array(3200).fill(0.1))],
      'utt.wav',
      { type: 'audio/wav' },
    )
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
      buttonNamed(host, 'Xác nhận & Hỏi trợ lý').click()
    })
    await flush()
    expect(session.executeChatTurn).toHaveBeenCalledTimes(1)
    expect(host.textContent).toMatch(/Không gửi được/)
    expect(host.querySelector('textarea[aria-label="Bản dịch tiếng Việt"]').value).toBe('tôi muốn khai sinh')
  })
})
