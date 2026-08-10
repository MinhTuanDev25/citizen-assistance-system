import { useEffect, useState } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { listCommunes } from '../api/catalog.js'
import { CommuneScene } from '../components/CommuneScene.jsx'
import { useCommune } from '../commune/CommuneContext.jsx'

export default function SelectCommunePage() {
  const { commune, setCommune } = useCommune()
  const navigate = useNavigate()
  const location = useLocation()
  const from = location.state?.from || '/'
  const changing = Boolean(location.state?.change)

  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      setLoading(true)
      setError('')
      try {
        const data = await listCommunes({ active: true })
        if (cancelled) return
        setItems(data?.items || [])
      } catch (err) {
        if (!cancelled) {
          setError(err.message || 'Không tải được danh sách xã')
          setItems([])
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  if (commune && !changing) {
    return <Navigate to={from === '/chon-xa' ? '/' : from} replace />
  }

  function pick(item) {
    setCommune({
      id: item.id,
      name: item.name,
      description: item.description || null,
    })
    navigate(from === '/chon-xa' ? '/' : from, { replace: true })
  }

  return (
    <div className="app-shell">
      <main className="commune-pick-wrap">
        <div className="commune-pick-layout">
          <aside className="commune-pick-visual">
            <CommuneScene className="commune-scene" />
            <div className="commune-pick-copy">
              <p className="commune-pick-eyebrow">Trợ lý thủ tục hành chính</p>
              <h2>Hỏi giấy tờ · điều kiện · nơi nộp</h2>
              <p>
                Hệ thống hướng dẫn công dân theo đúng địa bàn xã — nhanh, rõ,
                bằng tiếng Việt.
              </p>
              <ul className="commune-pick-points">
                <li>Không cần đăng nhập để hỏi thủ tục</li>
                <li>Danh mục theo xã bạn chọn</li>
                <li>Gợi ý giấy tờ và nơi nộp một cửa</li>
              </ul>
            </div>
          </aside>

          <div className="commune-pick-card">
            <h1>Bạn đang ở xã nào?</h1>
            <p className="muted">
              Chọn xã để xem đúng thủ tục tại địa phương của bạn.
            </p>

            {loading ? <p className="muted">Đang tải danh sách xã…</p> : null}
            {error ? (
              <p className="form-error">
                {error}. Kiểm tra API đang chạy rồi thử lại.
              </p>
            ) : null}

            <ul className="commune-pick-list">
              {items.map((item) => (
                <li key={item.id}>
                  <button type="button" onClick={() => pick(item)}>
                    <strong>Xã {item.name}</strong>
                    {item.description ? <span>{item.description}</span> : null}
                  </button>
                </li>
              ))}
            </ul>

            {!loading && !error && items.length === 0 ? (
              <p className="muted">Chưa có xã nào trong hệ thống.</p>
            ) : null}
          </div>
        </div>
      </main>
    </div>
  )
}
