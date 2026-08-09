import { useEffect, useRef, useState } from 'react'
import { sendTurn, WELCOME } from '../api/chat.js'
import { CitizenShell } from '../components/CitizenShell.jsx'
import { useAuth } from '../auth/AuthContext.jsx'

const SUGGESTIONS = [
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
  const [messages, setMessages] = useState([
    { id: 'welcome', role: 'assistant', text: WELCOME },
  ])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  async function submit(text) {
    const content = text.trim()
    if (!content || busy) return

    setMessages((prev) => [
      ...prev,
      { id: `u-${Date.now()}`, role: 'user', text: content },
    ])
    setDraft('')
    setBusy(true)

    try {
      const reply = await sendTurn({
        message: content,
        historyLength: messages.length + 1,
      })
      setMessages((prev) => [
        ...prev,
        { id: `a-${Date.now()}`, role: 'assistant', text: reply.text },
      ])
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: `e-${Date.now()}`,
          role: 'assistant',
          text: 'Xin lỗi, không gửi được tin nhắn. Vui lòng thử lại.',
        },
      ])
    } finally {
      setBusy(false)
      inputRef.current?.focus()
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit(draft)
    }
  }

  return (
    <CitizenShell>
      <main className="main">
        <section className="hero">
          <h1>Hỏi thủ tục hành chính bằng tiếng Việt</h1>
          <p>
            Tra cứu giấy tờ, điều kiện và nơi nộp tại xã
            {user ? ` · Xin chào, ${user.name}` : ' — không bắt buộc đăng nhập'}.
          </p>
          <div className="hints">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                type="button"
                className="hint"
                onClick={() => submit(s)}
                disabled={busy}
              >
                {s}
              </button>
            ))}
          </div>
        </section>

        <section className="chat-panel" aria-label="Hội thoại">
          <div className="chat-header">
            <div>
              <h2>Hội thoại hỗ trợ</h2>
              <p>
                {user ? `Phiên ${user.role}` : 'Phiên khách'} · demo mock
              </p>
            </div>
            <div className="status-dot" title="Sẵn sàng" />
          </div>

          <div className="messages" role="log" aria-live="polite">
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
              placeholder="Nhập câu hỏi của bạn…"
              rows={2}
              disabled={busy}
              aria-label="Nội dung câu hỏi"
            />
            <button type="submit" disabled={busy || !draft.trim()}>
              Gửi
            </button>
          </form>
        </section>
      </main>
    </CitizenShell>
  )
}
