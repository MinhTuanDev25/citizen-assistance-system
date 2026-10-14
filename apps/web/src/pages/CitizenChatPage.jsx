import { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { WELCOME } from '../api/chat.js'
import { listProcedures } from '../api/catalog.js'
import {
  ensureSession,
  executeChatTurn,
  listSessionMessages,
  orderedMessages,
  startFreshSession,
} from '../api/session.js'
import { CitizenLayout } from '../components/CitizenLayout.jsx'
import { useAuth } from '../auth/AuthContext.jsx'
import { useCommune } from '../commune/CommuneContext.jsx'

function Typing() {
  return (
    <span className="typing" aria-label="Đang trả lời">
      <i />
      <i />
      <i />
    </span>
  )
}

function ConfirmActions({ candidates, onPick, disabled }) {
  if (!candidates?.length) return null
  return (
    <div className="confirm-actions" role="group" aria-label="Xác nhận thủ tục">
      <button
        type="button"
        className="hint"
        disabled={disabled}
        onClick={() => onPick('Có')}
      >
        Có · {candidates[0].name || candidates[0].procedure_code}
      </button>
      <button
        type="button"
        className="hint"
        disabled={disabled}
        onClick={() => onPick('Không')}
      >
        Không
      </button>
      {candidates.slice(1).map((c) => (
        <button
          key={c.procedure_code}
          type="button"
          className="hint"
          disabled={disabled}
          onClick={() => onPick(c.name)}
        >
          {c.name}
        </button>
      ))}
    </div>
  )
}

export default function CitizenChatPage() {
  const { user } = useAuth()
  const { xaId, commune } = useCommune()
  const location = useLocation()
  const navigate = useNavigate()
  const askedRef = useRef('')
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [sessionReady, setSessionReady] = useState(false)
  const [sessionError, setSessionError] = useState('')
  const [chatSession, setChatSession] = useState(null)
  const [catalog, setCatalog] = useState([])
  const [catalogError, setCatalogError] = useState('')
  const [pendingConfirm, setPendingConfirm] = useState(null)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const messagesRef = useRef(null)
  const composingRef = useRef(false)
  const sentEchoRef = useRef({ text: '', at: 0 })
  const recognitionRef = useRef(null)
  const [listening, setListening] = useState(false)
  const [voiceHint, setVoiceHint] = useState('')

  useEffect(() => {
    const el = messagesRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [messages, busy, pendingConfirm])

  useEffect(() => {
    if (!sessionReady || busy) return
    inputRef.current?.focus()
  }, [sessionReady, busy])

  useEffect(() => {
    if (!xaId) return
    let cancelled = false
    ;(async () => {
      try {
        const data = await listProcedures({ xaId, citizen: true })
        if (cancelled) return
        setCatalog(data?.domains || [])
        setCatalogError('')
      } catch (err) {
        if (!cancelled) {
          setCatalog([])
          setCatalogError(err.message || 'Không tải được danh mục thủ tục')
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [xaId])

  useEffect(() => {
    if (!xaId) return
    let cancelled = false
    ;(async () => {
      setSessionReady(false)
      setSessionError('')
      try {
        const sess = await ensureSession(xaId)
        if (cancelled) return
        setChatSession(sess)

        const data = await listSessionMessages(sess.sessionId, sess.guestToken)
        if (cancelled) return
        const items = orderedMessages(data?.items)
        setMessages(
          items.length
            ? items
            : [{ id: 'welcome', role: 'assistant', text: WELCOME }],
        )
        const last = [...items].reverse().find((m) => m.role === 'assistant')
        setPendingConfirm(
          last?.action === 'CONFIRM_INTENT' ? last.candidates : null,
        )
        setSessionReady(true)
      } catch (err) {
        if (!cancelled) {
          setSessionError(err.message || 'Không tạo được phiên chat')
          setChatSession(null)
          setMessages([{ id: 'welcome', role: 'assistant', text: WELCOME }])
          setPendingConfirm(null)
          setSessionReady(false)
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [xaId, user?.id])

  async function startNewChat() {
    if (!xaId || busy) return
    setBusy(true)
    setSessionError('')
    try {
      const sess = await startFreshSession(xaId)
      setChatSession(sess)
      setDraft('')
      setMessages([{ id: 'welcome', role: 'assistant', text: WELCOME }])
      setPendingConfirm(null)
      setSessionReady(true)
      inputRef.current?.focus()
    } catch (err) {
      setSessionError(err.message || 'Không tạo được hội thoại mới')
    } finally {
      setBusy(false)
    }
  }

  async function submit(text) {
    const content = text.trim()
    if (!content || busy) return
    stopListening()

    setMessages((prev) => [
      ...prev,
      { id: `u-local-${Date.now()}`, role: 'user', text: content },
    ])
    sentEchoRef.current = { text: content, at: Date.now() }
    setDraft('')
    setBusy(true)
    setPendingConfirm(null)

    try {
      let sess = chatSession
      if (!sess?.sessionId) {
        sess = await ensureSession(xaId)
        setChatSession(sess)
        setSessionReady(true)
        setSessionError('')
      }

      const result = await executeChatTurn({
        sessionId: sess.sessionId,
        message: content,
        guestToken: sess.guestToken,
      })
      if (!result.ok) {
        throw result.error || new Error('Không gửi được tin nhắn')
      }
      if (result.historyFailed) {
        setSessionError(
          result.error?.message ||
            'Đã gửi tin nhưng không tải lại lịch sử. Thử làm mới trang.',
        )
      } else {
        const fromApi = orderedMessages(result.history?.items)
        setMessages(
          fromApi.length
            ? fromApi
            : [{ id: 'welcome', role: 'assistant', text: WELCOME }],
        )
        const last = [...fromApi].reverse().find((m) => m.role === 'assistant')
        setPendingConfirm(
          last?.action === 'CONFIRM_INTENT' ? last.candidates : null,
        )
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `e-${Date.now()}`,
          role: 'assistant',
          text:
            err.message ||
            'Xin lỗi, không gửi được tin nhắn. Kiểm tra API đang chạy.',
        },
      ])
    } finally {
      setBusy(false)
      inputRef.current?.focus()
    }
  }

  function stopListening() {
    try {
      recognitionRef.current?.stop()
    } catch {
      /* already stopped */
    }
    recognitionRef.current = null
    setListening(false)
  }

  useEffect(() => () => stopListening(), [])

  function toggleVoice() {
    const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!Ctor) {
      setVoiceHint('Trình duyệt này chưa hỗ trợ nhận giọng nói.')
      return
    }
    if (listening) {
      stopListening()
      return
    }
    setVoiceHint('')
    const rec = new Ctor()
    rec.lang = 'vi-VN'
    rec.interimResults = true
    rec.continuous = false
    rec.onresult = (event) => {
      let text = ''
      for (let i = 0; i < event.results.length; i += 1) {
        text += event.results[i][0].transcript
      }
      setDraft(text.trim())
    }
    rec.onerror = (event) => {
      if (event.error === 'not-allowed') {
        setVoiceHint('Cần cho phép micro để nói câu hỏi.')
      } else if (event.error !== 'aborted' && event.error !== 'no-speech') {
        setVoiceHint('Không nhận được giọng nói. Thử lại.')
      }
      stopListening()
    }
    rec.onend = () => {
      recognitionRef.current = null
      setListening(false)
    }
    recognitionRef.current = rec
    try {
      rec.start()
      setListening(true)
    } catch {
      setVoiceHint('Không bật được micro. Thử lại.')
      stopListening()
    }
  }

  function askAboutProcedure(proc) {
    submit(`Tôi muốn hỏi về thủ tục ${proc.name}`)
  }

  function onDraftChange(e) {
    const next = e.target.value
    const sent = sentEchoRef.current
    if (
      sent.text &&
      Date.now() - sent.at < 400 &&
      next &&
      (next === sent.text || sent.text.endsWith(next))
    ) {
      sentEchoRef.current = { text: '', at: 0 }
      return
    }
    if (!next) sentEchoRef.current = { text: '', at: 0 }
    setDraft(next)
  }

  function onKeyDown(e) {
    if (composingRef.current || e.nativeEvent.isComposing || e.keyCode === 229) {
      return
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit(draft)
    }
  }

  const suggestions = catalog
    .flatMap((g) => g.procedures || [])
    .slice(0, 4)

  useEffect(() => {
    const ask = location.state?.ask
    if (!ask || !sessionReady || busy) return
    if (askedRef.current === ask) return
    askedRef.current = ask
    navigate('/hoi', { replace: true, state: {} })
    void submit(ask)
  }, [busy, location.state, navigate, sessionReady])

  return (
    <CitizenLayout>
      <main className="citizen-main chat-only">
        <div className="quick-row" style={{ marginBottom: '0.75rem' }}>
          {suggestions.map((p) => (
            <button
              key={p.id}
              type="button"
              disabled={busy || !sessionReady}
              onClick={() => askAboutProcedure(p)}
            >
              {p.name}
            </button>
          ))}
        </div>
        {catalogError ? <p className="form-error">{catalogError}</p> : null}

        <section className="chat-panel chat-panel-sticky" aria-label="Hội thoại">
          <div className="chat-header">
            <div>
              <h2>Hỏi đáp thủ tục</h2>
              <p>
                {user ? `Xin chào, ${user.name}` : 'Phiên khách'} · xã{' '}
                {commune?.name}
                {sessionReady ? ' · đã nối API' : ''}
              </p>
            </div>
            <div className="chat-header-actions">
              <button
                type="button"
                className="new-chat"
                onClick={startNewChat}
                disabled={busy || !sessionReady}
              >
                Hội thoại mới
              </button>
              <div className="status-dot" title="Sẵn sàng" />
            </div>
          </div>

          {sessionError ? (
            <p className="catalog-error" style={{ margin: '0.75rem 1rem 0' }}>
              {sessionError}. Chạy <code>make api-run</code>.
            </p>
          ) : null}

          <div
            className="messages"
            ref={messagesRef}
            role="log"
            aria-live="polite"
          >
            {messages.map((m) => (
              <div
                key={m.id}
                className={`msg ${m.role === 'user' ? 'user' : 'bot'}`}
              >
                {m.text}
              </div>
            ))}
            {pendingConfirm ? (
              <ConfirmActions
                candidates={pendingConfirm}
                disabled={busy || !sessionReady}
                onPick={submit}
              />
            ) : null}
            {busy && (
              <div className="msg bot">
                <Typing />
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault()
              submit(draft)
            }}
          >
            <textarea
              ref={inputRef}
              value={draft}
              onChange={onDraftChange}
              onCompositionStart={() => {
                composingRef.current = true
              }}
              onCompositionEnd={() => {
                composingRef.current = false
              }}
              onKeyDown={onKeyDown}
              placeholder={
                listening
                  ? 'Đang nghe… hãy nói câu hỏi'
                  : sessionReady
                    ? 'Nhập hoặc nói câu hỏi…'
                    : 'Đang mở phiên chat…'
              }
              rows={2}
              disabled={!sessionReady}
              spellCheck={false}
              aria-label="Nội dung câu hỏi"
            />
            <button
              type="button"
              className={`voice-mic${listening ? ' listening' : ''}`}
              onClick={toggleVoice}
              disabled={busy || !sessionReady}
              aria-pressed={listening}
              aria-label={listening ? 'Dừng nghe' : 'Nói câu hỏi'}
              title={listening ? 'Dừng nghe' : 'Nói câu hỏi'}
            >
              {listening ? '●' : '🎙'}
            </button>
            <button
              type="submit"
              className="composer-send"
              onMouseDown={(e) => e.preventDefault()}
              disabled={busy || !sessionReady || !draft.trim()}
            >
              Gửi
            </button>
          </form>
          {voiceHint ? (
            <p className="form-error" style={{ margin: '0 0.9rem 0.75rem' }}>
              {voiceHint}
            </p>
          ) : null}
        </section>
      </main>
    </CitizenLayout>
  )
}
