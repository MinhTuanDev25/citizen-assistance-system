import { useEffect, useState } from 'react'
import { Link, Navigate, useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext.jsx'
import { CitizenShell } from '../components/CitizenShell.jsx'

const PRESETS = {
  citizen: {
    email: 'citizen@example.com',
    password: 'citizen123',
    title: 'Đăng nhập công dân',
    blurb:
      'Đăng nhập để lưu phiên hỏi thủ tục. Bạn vẫn có thể hỏi như khách nếu chưa có tài khoản.',
  },
  admin: {
    email: 'admin@chuse.vn',
    password: 'admin123',
    title: 'Đăng nhập cán bộ',
    blurb:
      'Tài khoản Admin dùng để upload tài liệu, duyệt bản nháp và publish thủ tục.',
  },
}

export default function LoginPage() {
  const { login, user } = useAuth()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const mode = searchParams.get('as') === 'admin' ? 'admin' : 'citizen'
  const preset = PRESETS[mode]

  const [email, setEmail] = useState(PRESETS.citizen.email)
  const [password, setPassword] = useState(PRESETS.citizen.password)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    setEmail(PRESETS[mode].email)
    setPassword(PRESETS[mode].password)
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

          <h1>{preset.title}</h1>
          <p className="muted">{preset.blurb}</p>

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
            {mode === 'citizen' ? (
              <p>
                <strong>Demo API:</strong> citizen@example.com / citizen123
              </p>
            ) : (
              <p>
                <strong>Demo API:</strong> admin@chuse.vn / admin123
              </p>
            )}
            <Link to="/">Tiếp tục hỏi thủ tục (khách)</Link>
          </div>
        </form>
      </main>
    </CitizenShell>
  )
}
