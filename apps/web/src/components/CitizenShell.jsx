import { Link, NavLink } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext.jsx'

export function CitizenShell({ children }) {
  const { user, logout } = useAuth()

  return (
    <div className="app-shell">
      <header className="topbar">
        <Link to="/" className="brand-mark brand-link">
          <div className="brand-icon" aria-hidden>
            CS
          </div>
          <div>
            <strong>Trợ lý thủ tục Chư Sê</strong>
            <span>Xã Chư Sê · Gia Lai</span>
          </div>
        </Link>
        <nav className="top-nav">
          <NavLink to="/" end>
            Hỏi thủ tục
          </NavLink>
          {user?.role === 'ADMIN' ? (
            <NavLink to="/admin">Admin</NavLink>
          ) : (
            <NavLink to="/login">Đăng nhập cán bộ</NavLink>
          )}
          {user ? (
            <button type="button" className="linkish" onClick={logout}>
              Đăng xuất
            </button>
          ) : null}
        </nav>
      </header>
      {children}
    </div>
  )
}
