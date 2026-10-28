import { useEffect, useState } from 'react'
import { Link, Navigate, useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext.jsx'
import { ROUTES } from '../routes.js'
import { ServiceMark, UiIcon } from '../components/UiIcon.jsx'

// Compile-time flag: production builds leave VITE_DEMO_LOGIN unset → dead branch tree-shaken.
const DEMO_ON =
  import.meta.env.VITE_DEMO_LOGIN === 'true' ||
  import.meta.env.VITE_DEMO_LOGIN === '1'

const DEMO_PRESETS = DEMO_ON
  ? {
      citizen: { email: 'citizen@example.com', password: 'citizen123' },
      admin: { email: 'admin@chuse.vn', password: 'admin123' },
    }
  : null

export default function LoginPage() {
  const { login, user } = useAuth()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const mode = searchParams.get('as') === 'admin' ? 'admin' : 'citizen'

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (DEMO_PRESETS) {
      setEmail(DEMO_PRESETS[mode].email)
      setPassword(DEMO_PRESETS[mode].password)
    } else {
      setEmail('')
      setPassword('')
    }
    setError('')
  }, [mode])

  if (user?.role === 'ADMIN') {
    return <Navigate to={ROUTES.admin} replace />
  }
  if (user?.role === 'CITIZEN') {
    return <Navigate to={ROUTES.home} replace />
  }

  function switchMode(next) {
    setSearchParams(next === 'admin' ? { as: 'admin' } : { as: 'citizen' })
  }

  async function onSubmit(e) {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      const next = await login(email, password)
      navigate(next.role === 'ADMIN' ? ROUTES.admin : ROUTES.home, { replace: true })
    } catch (err) {
      setError(err.message || 'Đăng nhập thất bại')
    } finally {
      setBusy(false)
    }
  }

  const title =
    mode === 'admin' ? 'Đăng nhập cán bộ' : 'Đăng nhập công dân'
  const blurb =
    mode === 'admin'
      ? 'Tài khoản cán bộ dùng để quản lý thủ tục. Hệ thống chính thức không sử dụng mật khẩu mặc định.'
      : 'Đăng nhập để lưu phiên hỏi thủ tục. Bạn vẫn có thể hỏi như khách nếu chưa có tài khoản.'

  return (
    <div className="login-screen">
      <aside className="login-hero">
        <div className="login-hero-brand">
          <ServiceMark />
          <span>Trợ lý hành chính</span>
        </div>
        <span className="login-hero-kicker"><UiIcon name="shield" size={18} /> Dịch vụ công trực tuyến</span>
        <h2>Công dân số phục vụ người dân</h2>
        <p>Hỏi đáp, tra cứu và chuẩn bị hồ sơ ngay tại xã, hoàn toàn bằng tiếng Việt.</p>
      </aside>
      <main className="login-panel">
        <form className="login-card" onSubmit={onSubmit}>
          <div className="login-tabs" role="tablist" aria-label="Loại tài khoản">
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'citizen'}
              className={mode === 'citizen' ? 'active' : ''}
              onClick={() => switchMode('citizen')}
            >
              Công dân
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'admin'}
              className={mode === 'admin' ? 'active' : ''}
              onClick={() => switchMode('admin')}
            >
              Cán bộ
            </button>
          </div>

          <h1>{title}</h1>
          <p className="muted">{blurb}</p>

          <label>
            Thư điện tử
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="username"
              placeholder="Nhập địa chỉ thư điện tử"
              required
            />
          </label>
          <label>
            Mật khẩu
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              placeholder="••••••••"
              required
            />
          </label>

          {error ? <p className="form-error">{error}</p> : null}

          <button type="submit" className="btn-primary" disabled={busy}>
            {busy ? 'Đang vào…' : 'Đăng nhập'}
          </button>

          <div className="login-hint">
            {DEMO_ON ? (
              <p className="cell-muted">
                Chế độ đăng nhập dùng thử đang được bật và không xuất hiện trên hệ thống chính thức.
              </p>
            ) : null}
            <Link to={ROUTES.home}>Tiếp tục hỏi thủ tục (khách)</Link>
          </div>
        </form>
      </main>
    </div>
  )
}
