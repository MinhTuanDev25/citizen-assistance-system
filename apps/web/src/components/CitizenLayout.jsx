import { Link, NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext.jsx'
import { useCommune } from '../commune/CommuneContext.jsx'

const NAV = [
  { to: '/', end: true, label: 'Trang chủ', icon: '⌂' },
  { to: '/hoi', label: 'Hỏi đáp', icon: '💬' },
  { to: '/thu-tuc', label: 'Thủ tục hành chính', icon: '☰' },
]

export function CitizenLayout({ children }) {
  const { user, logout } = useAuth()
  const { commune } = useCommune()
  const navigate = useNavigate()
  const initial = (user?.name || 'K').trim().charAt(0).toUpperCase()

  return (
    <div className="cas-layout">
      <aside className="cas-side">
        <Link to="/" className="brand-mark brand-link">
          <div className="brand-icon" aria-hidden>
            CS
          </div>
          <div>
            <strong>Citizen Assistance</strong>
            <span>Xã {commune?.name || '…'}</span>
          </div>
        </Link>
        <nav className="cas-side-nav" aria-label="Cổng công dân">
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end}>
              <span className="nav-ico" aria-hidden>
                {item.icon}
              </span>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <p className="cas-side-section">Tài khoản</p>
        <nav className="cas-side-nav">
          {user?.role === 'ADMIN' ? <NavLink to="/admin">Cổng cán bộ</NavLink> : null}
          {!user ? <NavLink to="/login?as=citizen">Đăng nhập</NavLink> : null}
          <button type="button" onClick={() => navigate('/chon-xa', { state: { change: true, from: '/' } })}>
            Đổi xã
          </button>
        </nav>
        <div className="cas-side-foot">
          {user ? (
            <button type="button" className="linkish dark" onClick={() => void logout()}>
              Đăng xuất
            </button>
          ) : (
            <span className="cell-muted">Đang dùng với tư cách khách</span>
          )}
        </div>
      </aside>
      <div className="cas-body">
        <header className="cas-top">
          <div className="cas-search">Tìm thủ tục, điều kiện…</div>
          <div className="cas-userchip">
            <span className="cas-avatar">{initial}</span>
            {user?.name || 'Khách'}
          </div>
        </header>
        <div className="cas-main">{children}</div>
      </div>
    </div>
  )
}
