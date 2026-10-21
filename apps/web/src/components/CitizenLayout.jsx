import { Link, NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext.jsx'
import { useCommune } from '../commune/CommuneContext.jsx'
import { ROUTES, loginPath } from '../routes.js'
import CitizenSearch from './CitizenSearch.jsx'

const NAV = [
  { to: ROUTES.home, end: true, label: 'Trang chủ', icon: '⌂' },
  { to: ROUTES.chat, label: 'Hỏi đáp', icon: '💬' },
  { to: ROUTES.procedures, label: 'Thủ tục hành chính', icon: '☰' },
]

export function CitizenLayout({ children }) {
  const { user, logout } = useAuth()
  const { commune } = useCommune()
  const navigate = useNavigate()
  const initial = (user?.name || 'K').trim().charAt(0).toUpperCase()

  return (
    <div className="cas-layout">
      <aside className="cas-side">
        <Link to={ROUTES.home} className="brand-mark brand-link">
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
          {user?.role === 'ADMIN' ? <NavLink to={ROUTES.admin}>Cổng cán bộ</NavLink> : null}
          {!user ? <NavLink to={loginPath('citizen')}>Đăng nhập</NavLink> : null}
          <button type="button" onClick={() => navigate(ROUTES.commune, { state: { change: true, from: ROUTES.home } })}>
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
          <CitizenSearch />
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
