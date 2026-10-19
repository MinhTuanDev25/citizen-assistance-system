import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { listProcedures } from '../../api/catalog.js'
import { listDocuments } from '../../api/documents.js'
import { useCommune } from '../../commune/CommuneContext.jsx'

const STEPS = [
  {
    to: '/admin/documents',
    step: '1',
    title: 'Tải PDF',
    text: 'Lưu tài liệu nguồn theo lĩnh vực.',
    bg: '#dbeafe',
  },
  {
    to: '/admin/documents',
    step: '2',
    title: 'Gắn và lập chỉ mục',
    text: 'Nối PDF với phiên bản thủ tục, chờ sẵn sàng.',
    bg: '#dcfce7',
  },
  {
    to: '/admin/procedures',
    step: '3',
    title: 'Kích hoạt phiên bản',
    text: 'Công dân hỏi đáp theo phiên đang dùng.',
    bg: '#ffedd5',
  },
]

export default function AdminDashboardPage() {
  const { xaId, commune } = useCommune()
  const [docs, setDocs] = useState(null)
  const [active, setActive] = useState(null)
  const [apiError, setApiError] = useState('')

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const data = await listDocuments({})
        if (!cancelled) setDocs(data?.count ?? 0)
      } catch (err) {
        if (!cancelled) setApiError(err.message || 'API tài liệu lỗi')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!xaId) return
    let cancelled = false
    ;(async () => {
      try {
        const data = await listProcedures({ xaId, citizen: false })
        if (!cancelled) setActive(data?.count ?? 0)
      } catch (err) {
        if (!cancelled) setApiError(err.message || 'Không tải được thủ tục')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [xaId])

  return (
    <div className="admin-page">
      <header className="admin-page-head">
        <div>
          <h1>Tổng quan</h1>
          <p>Xã {commune?.name || '…'}</p>
        </div>
      </header>

      {apiError ? <p className="form-error">{apiError}</p> : null}

      <div className="stat-grid stat-grid-2">
        <div className="stat-card accent-blue">
          <span>Tài liệu PDF</span>
          <strong>{docs == null ? '…' : docs}</strong>
          <Link to="/admin/documents">Mở tài liệu</Link>
        </div>
        <div className="stat-card accent-teal">
          <span>Thủ tục đang dùng</span>
          <strong>{active == null ? '…' : active}</strong>
          <Link to="/admin/procedures">Mở thủ tục</Link>
        </div>
      </div>

      <p className="cas-side-section">Luồng làm việc</p>
      <div className="home-grid admin-flow">
        {STEPS.map((card) => (
          <Link key={card.title} to={card.to} className="home-card">
            <div className="icon" style={{ background: card.bg }} aria-hidden>
              {card.step}
            </div>
            <strong>{card.title}</strong>
            <span>{card.text}</span>
          </Link>
        ))}
      </div>
    </div>
  )
}
