import { Link, NavLink, Outlet } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext.jsx'
import { useCommune } from '../commune/CommuneContext.jsx'
import { adminIngestionEnabled } from '../config/features.js'

const ingestionOn = adminIngestionEnabled()

const LINKS = [
  { to: '/admin', end: true, label: 'Tổng quan', icon: '⌂' },
  { to: '/admin/documents', label: ingestionOn ? 'Tài liệu' : 'Tài liệu', icon: '📄' },
  { to: '/admin/procedures', label: 'Thủ tục', icon: '☰' },
  { to: '/admin/drafts', label: 'Bản nháp', icon: '✎', soon: true },
]

export function AdminShell() {
  const { user, logout } = useAuth()
  const { commune } = useCommune()
  const initial = (user?.name || 'A').trim().charAt(0).toUpperCase()

  return (
    <div className="cas-layout">
      <aside className="cas-side">
        <Link to="/admin" className="brand-mark brand-link">
          <div className="brand-icon" aria-hidden>
            CS
          </div>
          <div>
            <strong>Citizen Assistance</strong>
            <span>Cổng cán bộ · {commune?.name || '…'}</span>
          </div>
        </Link>
        <nav className="cas-side-nav" aria-label="Cổng cán bộ">
          {LINKS.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end} className={item.soon ? 'soon' : undefined}>
              <span className="nav-ico" aria-hidden>
                {item.icon}
              </span>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <p className="cas-side-section">Khác</p>
        <nav className="cas-side-nav">
          <NavLink to="/">Cổng công dân</NavLink>
        </nav>
        <div className="cas-side-foot">
          <button type="button" className="linkish dark" onClick={() => void logout()}>
            Đăng xuất
          </button>
        </div>
      </aside>
      <div className="cas-body">
        <header className="cas-top">
          <div className="cas-search">Tìm tài liệu, thủ tục…</div>
          <div className="cas-userchip">
            <span className="cas-avatar">{initial}</span>
            {user?.name || 'Cán bộ'}
          </div>
        </header>
        <div className="cas-main">
          <Outlet />
        </div>
      </div>
    </div>
  )
}
