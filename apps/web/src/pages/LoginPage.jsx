import { useEffect, useState } from 'react'
import { Link, Navigate, useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext.jsx'
import { CitizenShell } from '../components/CitizenShell.jsx'

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
    return <Navigate to="/admin" replace />
  }
  if (user?.role === 'CITIZEN') {
    return <Navigate to="/" replace />
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
      navigate(next.role === 'ADMIN' ? '/admin' : '/', { replace: true })
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
      ? 'Tài khoản Admin dùng để quản lý thủ tục. Không có mật khẩu mặc định trên production.'
      : 'Đăng nhập để lưu phiên hỏi thủ tục. Bạn vẫn có thể hỏi như khách nếu chưa có tài khoản.'

  return (
    <CitizenShell>
      <main className="login-wrap">
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
            Email
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="username"
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
                Dev demo login enabled (VITE_DEMO_LOGIN). Not present in
                production builds.
              </p>
            ) : null}
            <Link to="/">Tiếp tục hỏi thủ tục (khách)</Link>
          </div>
        </form>
      </main>
    </CitizenShell>
  )
}
