import { Link, NavLink, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext.jsx'
import { useCommune } from '../commune/CommuneContext.jsx'

export function CitizenShell({ children }) {
  const { user, logout } = useAuth()
  const { commune } = useCommune()
  const navigate = useNavigate()
  const location = useLocation()
  const loginAs = new URLSearchParams(location.search).get('as')
  const onLogin = location.pathname === '/login'

  function changeCommune() {
    navigate('/chon-xa', { state: { change: true, from: '/' } })
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <Link to="/" className="brand-mark brand-link">
          <div className="brand-icon" aria-hidden>
            CS
          </div>
          <div>
            <strong>Trợ lý thủ tục</strong>
            <span>
              {commune ? `Xã ${commune.name}` : 'Chọn xã'} · Gia Lai
            </span>
          </div>
        </Link>
        <nav className="top-nav">
          <NavLink to="/" end>
            Hỏi thủ tục
          </NavLink>
          {commune ? (
            <button type="button" className="linkish" onClick={changeCommune}>
              Đổi xã
            </button>
          ) : null}
          {user?.role === 'ADMIN' ? <NavLink to="/admin">Admin</NavLink> : null}
          {!user ? (
            <>
              <Link
                to="/login?as=citizen"
                className={onLogin && loginAs !== 'admin' ? 'active' : undefined}
              >
                Đăng nhập công dân
              </Link>
              <Link
                to="/login?as=admin"
                className={onLogin && loginAs === 'admin' ? 'active' : undefined}
              >
                Đăng nhập cán bộ
              </Link>
            </>
          ) : (
            <span className="nav-user" title={user.email}>
              {user.name}
            </span>
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
