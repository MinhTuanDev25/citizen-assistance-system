import { Link, NavLink, Outlet } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext.jsx'
import { useCommune } from '../commune/CommuneContext.jsx'
import { adminIngestionEnabled } from '../config/features.js'
import { ROUTES } from '../routes.js'
import AdminSearch from './AdminSearch.jsx'

const ingestionOn = adminIngestionEnabled()

const LINKS = [
  { to: ROUTES.admin, end: true, label: 'Tổng quan', icon: '⌂' },
  { to: ROUTES.adminDocuments, label: ingestionOn ? 'Tài liệu' : 'Tài liệu', icon: '📄' },
  { to: ROUTES.adminProcedures, label: 'Thủ tục', icon: '☰' },
  { to: ROUTES.adminDrafts, label: 'Bản nháp', icon: '✎' },
]

export function AdminShell() {
  const { user, logout } = useAuth()
  const { commune } = useCommune()
  const initial = (user?.name || 'A').trim().charAt(0).toUpperCase()

  return (
    <div className="cas-layout">
      <aside className="cas-side">
        <Link to={ROUTES.admin} className="brand-mark brand-link">
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
          <NavLink to={ROUTES.home}>Cổng công dân</NavLink>
        </nav>
        <div className="cas-side-foot">
          <button type="button" className="linkish dark" onClick={() => void logout()}>
            Đăng xuất
          </button>
        </div>
      </aside>
      <div className="cas-body">
        <header className="cas-top">
          <AdminSearch />
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
