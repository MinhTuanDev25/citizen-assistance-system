import { useState } from 'react'
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext.jsx'
import { CitizenShell } from '../components/CitizenShell.jsx'

export default function LoginPage() {
  const { login, user } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const from = location.state?.from || '/admin'

  const [email, setEmail] = useState('admin@chuse.vn')
  const [password, setPassword] = useState('admin123')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  if (user?.role === 'ADMIN') {
    return <Navigate to="/admin" replace />
  }

  function onSubmit(e) {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      const next = login(email, password)
      navigate(next.role === 'ADMIN' ? from : '/', { replace: true })
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
          <h1>Đăng nhập</h1>
          <p className="muted">
            Cán bộ dùng tài khoản Admin để upload / duyệt / publish thủ tục.
            Công dân V1 có thể hỏi thủ tục không cần đăng nhập.
          </p>

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
            <p>
              <strong>Admin demo:</strong> admin@chuse.vn / admin123
            </p>
            <p>
              <strong>Citizen demo:</strong> citizen@example.com / citizen123
            </p>
            <Link to="/">Tiếp tục hỏi thủ tục (khách)</Link>
          </div>
        </form>
      </main>
    </CitizenShell>
  )
}
