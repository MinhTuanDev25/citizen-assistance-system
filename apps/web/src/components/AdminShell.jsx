import { Link, NavLink, Outlet } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext.jsx'

const LINKS = [
  { to: '/admin', end: true, label: 'Tổng quan' },
  { to: '/admin/documents', label: 'Tài liệu' },
  { to: '/admin/drafts', label: 'Bản nháp' },
  { to: '/admin/procedures', label: 'Thủ tục' },
]

export function AdminShell() {
  const { user, logout } = useAuth()

  return (
    <div className="admin-shell">
      <aside className="admin-side">
        <Link to="/admin" className="brand-mark brand-link admin-brand">
          <div className="brand-icon" aria-hidden>
            CS
          </div>
          <div>
            <strong>Admin Chư Sê</strong>
            <span>Knowledge pipeline</span>
          </div>
        </Link>
        <nav className="admin-nav">
          {LINKS.map((l) => (
            <NavLink key={l.to} to={l.to} end={l.end}>
              {l.label}
            </NavLink>
          ))}
        </nav>
        <div className="admin-side-foot">
          <p>{user?.name}</p>
          <Link to="/">← Cổng công dân</Link>
          <button type="button" className="linkish" onClick={logout}>
            Đăng xuất
          </button>
        </div>
      </aside>
      <div className="admin-main">
        <Outlet />
      </div>
    </div>
  )
}
