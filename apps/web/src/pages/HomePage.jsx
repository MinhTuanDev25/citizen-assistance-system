import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { listProcedures } from '../api/catalog.js'
import { CitizenLayout } from '../components/CitizenLayout.jsx'
import { useAuth } from '../auth/AuthContext.jsx'
import { useCommune } from '../commune/CommuneContext.jsx'
import { ROUTES } from '../routes.js'

const CARDS = [
  { to: ROUTES.chat, title: 'Hỏi về thủ tục', text: 'Gõ hoặc nói để hỏi giấy tờ, phí, hạn và nơi nộp.', icon: '💬', bg: '#dbeafe' },
  { to: ROUTES.procedures, title: 'Hướng dẫn hồ sơ', text: 'Danh mục thủ tục đang hỗ trợ tại xã.', icon: '📁', bg: '#dcfce7' },
]

export default function HomePage() {
  const { user } = useAuth()
  const { xaId, commune } = useCommune()
  const navigate = useNavigate()
  const [items, setItems] = useState([])

  useEffect(() => {
    if (!xaId) return
    let cancelled = false
    ;(async () => {
      try {
        const data = await listProcedures({ xaId, citizen: true })
        if (cancelled) return
        const procs = (data?.domains || []).flatMap((g) => g.procedures || [])
        setItems(procs.slice(0, 6))
      } catch {
        if (!cancelled) setItems([])
      }
    })()
    return () => {
      cancelled = true
    }
  }, [xaId])

  return (
    <CitizenLayout>
      <section className="home-hero">
        <div>
          <h1>Xin chào{user?.name ? `, ${user.name}` : ''}!</h1>
          <p>
            Tôi là trợ lý AI hỗ trợ thủ tục hành chính của xã {commune?.name || '…'}.
            Bạn cần tôi giúp gì hôm nay?
          </p>
        </div>
        <div className="home-bot" aria-hidden>
          <svg viewBox="0 0 160 140" width="160" height="140">
            <ellipse cx="80" cy="128" rx="42" ry="8" fill="#93c5fd" opacity="0.45" />
            <rect x="48" y="38" width="64" height="58" rx="18" fill="#60a5fa" />
            <rect x="56" y="48" width="48" height="32" rx="12" fill="#eff6ff" />
            <circle cx="70" cy="64" r="5" fill="#1d4ed8" />
            <circle cx="90" cy="64" r="5" fill="#1d4ed8" />
            <rect x="72" y="78" width="16" height="4" rx="2" fill="#1d4ed8" />
            <rect x="70" y="18" width="20" height="16" rx="8" fill="#93c5fd" />
            <circle cx="80" cy="14" r="6" fill="#38bdf8" />
            <rect x="32" y="58" width="12" height="28" rx="6" fill="#93c5fd" />
            <rect x="116" y="58" width="12" height="28" rx="6" fill="#93c5fd" />
          </svg>
        </div>
      </section>
      <div className="home-grid">
        {CARDS.map((card) => (
          <Link key={card.to} to={card.to} className="home-card">
            <div className="icon" style={{ background: card.bg }} aria-hidden>
              {card.icon}
            </div>
            <strong>{card.title}</strong>
            <span>{card.text}</span>
          </Link>
        ))}
      </div>
      <p className="cas-side-section">Thủ tục hay dùng</p>
      <div className="quick-row">
        {items.map((p) => (
          <button
            key={p.id}
            type="button"
            onClick={() => navigate(ROUTES.chat, { state: { ask: `Tôi muốn hỏi về thủ tục ${p.name}` } })}
          >
            {p.name}
          </button>
        ))}
        <Link to={ROUTES.procedures}>Xem thêm</Link>
      </div>
    </CitizenLayout>
  )
}
