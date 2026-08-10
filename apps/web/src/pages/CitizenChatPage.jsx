import { useEffect, useRef, useState } from 'react'
import { sendTurn, WELCOME } from '../api/chat.js'
import { listProcedures } from '../api/catalog.js'
import {
  ensureSession,
  listSessionMessages,
  mapApiMessage,
  postUserMessage,
} from '../api/session.js'
import { CitizenShell } from '../components/CitizenShell.jsx'
import { CommuneScene } from '../components/CommuneScene.jsx'
import { useAuth } from '../auth/AuthContext.jsx'
import { useCommune } from '../commune/CommuneContext.jsx'

const FALLBACK_SUGGESTIONS = [
  'Đăng ký khai sinh cần gì?',
  'Chứng thực giấy tờ như thế nào?',
  'Giờ làm việc bộ phận một cửa?',
]

function Typing() {
  return (
    <span className="typing" aria-label="Đang trả lời">
      <i />
      <i />
      <i />
    </span>
  )
}

export default function CitizenChatPage() {
  const { user } = useAuth()
  const { xaId, commune } = useCommune()
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [sessionReady, setSessionReady] = useState(false)
  const [sessionError, setSessionError] = useState('')
  const [chatSession, setChatSession] = useState(null)
  const [catalog, setCatalog] = useState([])
  const [catalogError, setCatalogError] = useState('')
  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const messagesRef = useRef(null)

  useEffect(() => {
    const el = messagesRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [messages, busy])

  useEffect(() => {
    if (!xaId) return
    let cancelled = false
    ;(async () => {
      try {
        const data = await listProcedures({ xaId })
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
        const items = (data?.items || []).map(mapApiMessage)
        setMessages(
          items.length
            ? items
            : [{ id: 'welcome', role: 'assistant', text: WELCOME }],
        )
        setSessionReady(true)
      } catch (err) {
        if (!cancelled) {
          setSessionError(err.message || 'Không tạo được phiên chat')
          setChatSession(null)
          setMessages([{ id: 'welcome', role: 'assistant', text: WELCOME }])
          setSessionReady(false)
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [xaId, user?.id])

  async function submit(text) {
    const content = text.trim()
    if (!content || busy) return

    setMessages((prev) => [
      ...prev,
      { id: `u-local-${Date.now()}`, role: 'user', text: content },
    ])
    setDraft('')
    setBusy(true)

    try {
      let sess = chatSession
      if (!sess?.sessionId) {
        sess = await ensureSession(xaId)
        setChatSession(sess)
        setSessionReady(true)
        setSessionError('')
      }

      await postUserMessage(sess.sessionId, content, sess.guestToken)

      const history = await listSessionMessages(sess.sessionId, sess.guestToken)
      const fromApi = (history?.items || []).map(mapApiMessage)

      // Decision Engine chưa nối — trả lời tạm bằng mock sau khi đã persist USER
      const reply = await sendTurn({
        message: content,
        historyLength: fromApi.length + 1,
      })

      setMessages([
        ...fromApi,
        {
          id: `a-local-${Date.now()}`,
          role: 'assistant',
          text: reply.text,
        },
      ])
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

  function askAboutProcedure(proc) {
    submit(`Tôi muốn hỏi về thủ tục ${proc.name}`)
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit(draft)
    }
  }

  const suggestions =
    catalog.length > 0
      ? catalog
          .flatMap((g) => g.procedures || [])
          .slice(0, 4)
          .map((p) => p.name)
      : FALLBACK_SUGGESTIONS

  return (
    <CitizenShell>
      <main className="citizen-main">
        <section className="citizen-intro" aria-label="Giới thiệu">
          <div className="citizen-intro-copy">
            <p className="citizen-intro-eyebrow">Trợ lý thủ tục hành chính</p>
            <h1>Hỏi thủ tục bằng tiếng Việt</h1>
            <p>
              Tra cứu giấy tờ, điều kiện và nơi nộp tại{' '}
              <strong>xã {commune?.name || '…'}</strong>
              {user ? ` · Xin chào, ${user.name}` : ''}. Không bắt buộc đăng
              nhập.
            </p>
            <div className="hints">
              {suggestions.map((s) => (
                <button
                  key={s}
                  type="button"
                  className="hint"
                  spellCheck={false}
                  onClick={() => submit(s)}
                  disabled={busy || !sessionReady}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
          <div className="citizen-intro-visual">
            <CommuneScene className="commune-scene" />
          </div>
        </section>

        <div className="citizen-aside">
          <section className="catalog-panel" aria-label="Danh mục thủ tục">
            <div className="catalog-head">
              <h2>Thủ tục đang hỗ trợ</h2>
              <p>Bấm để hỏi nhanh theo danh mục của xã</p>
            </div>
            {catalogError ? (
              <p className="catalog-error">{catalogError}</p>
            ) : null}
            {!catalogError && catalog.length === 0 ? (
              <p className="catalog-muted">Đang tải danh mục…</p>
            ) : null}
            <div className="catalog-groups">
              {catalog.map((g) => (
                <div key={g.domain_id} className="catalog-group">
                  <h3>
                    {g.domain_name} <span>{g.count}</span>
                  </h3>
                  <ul>
                    {(g.procedures || []).map((p) => (
                      <li key={p.id}>
                        <button
                          type="button"
                          spellCheck={false}
                          onClick={() => askAboutProcedure(p)}
                          disabled={busy || !sessionReady}
                        >
                          <strong>{p.name}</strong>
                          <span>Hỏi hướng dẫn</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>
        </div>

        <section className="chat-panel chat-panel-sticky" aria-label="Hội thoại">
          <div className="chat-header">
            <div>
              <h2>Hội thoại hỗ trợ</h2>
              <p>
                {user ? `Xin chào, ${user.name}` : 'Phiên khách'} · xã{' '}
                {commune?.name}
                {sessionReady ? ' · đã nối API' : ''}
              </p>
            </div>
            <div className="status-dot" title="Sẵn sàng" />
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
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder={
                sessionReady
                  ? 'Nhập câu hỏi của bạn…'
                  : 'Đang mở phiên chat…'
              }
              rows={2}
              disabled={busy || !sessionReady}
              spellCheck={false}
              aria-label="Nội dung câu hỏi"
            />
            <button
              type="submit"
              disabled={busy || !sessionReady || !draft.trim()}
            >
              Gửi
            </button>
          </form>
        </section>
      </main>
    </CitizenShell>
  )
}
