import { useState } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext.jsx'
import { ROUTES, loginPath } from '../routes.js'
import { ServiceMark, UiIcon } from '../components/UiIcon.jsx'

// Same limits as POST /api/v1/auth/register, checked here so the citizen sees them first.
export function validateRegistration({ fullName, email, password, confirm }) {
  if ([...fullName.trim()].length < 2) return 'Nhập họ tên (ít nhất 2 ký tự).'
  if (!/^[^\s@]+@[^\s@]+$/.test(email.trim())) return 'Email không hợp lệ.'
  if (password.length < 6) return 'Mật khẩu phải có ít nhất 6 ký tự.'
  if (password !== confirm) return 'Mật khẩu nhập lại không khớp.'
  return ''
}

export default function RegisterPage() {
  const { register, user } = useAuth()
  const navigate = useNavigate()
  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  if (user?.role === 'ADMIN') return <Navigate to={ROUTES.admin} replace />
  if (user?.role === 'CITIZEN') return <Navigate to={ROUTES.home} replace />

  async function onSubmit(e) {
    e.preventDefault()
    const invalid = validateRegistration({ fullName, email, password, confirm })
    if (invalid) {
      setError(invalid)
      return
    }
    setError('')
    setBusy(true)
    try {
      await register({ fullName: fullName.trim(), email: email.trim(), password })
      navigate(ROUTES.home, { replace: true })
    } catch (err) {
      setError(err.message || 'Không đăng ký được. Thử lại sau.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="login-screen">
      <aside className="login-hero">
        <div className="login-hero-brand">
          <ServiceMark />
          <span>Trợ lý hành chính</span>
        </div>
        <span className="login-hero-kicker"><UiIcon name="shield" size={18} /> Dịch vụ công trực tuyến</span>
        <h2>Công dân số phục vụ người dân</h2>
        <p>Tạo tài khoản để lưu các phiên hỏi thủ tục của bạn.</p>
      </aside>
      <main className="login-panel">
        <form className="login-card" onSubmit={onSubmit} noValidate>
          <h1>Đăng ký tài khoản công dân</h1>
          <p className="muted">Tài khoản cán bộ do quản trị viên cấp, không đăng ký tại đây.</p>

          <label>
            Họ và tên
            <input value={fullName} onChange={(e) => setFullName(e.target.value)} autoComplete="name" placeholder="Nguyễn Văn A" required />
          </label>
          <label>
            Thư điện tử
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" placeholder="Nhập địa chỉ thư điện tử" required />
          </label>
          <label>
            Mật khẩu
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="new-password" placeholder="Ít nhất 6 ký tự" required />
          </label>
          <label>
            Nhập lại mật khẩu
            <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} autoComplete="new-password" placeholder="••••••••" required />
          </label>

          {error ? <p className="form-error" role="alert">{error}</p> : null}

          <button type="submit" className="btn-primary" disabled={busy}>
            {busy ? 'Đang đăng ký…' : 'Đăng ký'}
          </button>

          <div className="login-hint">
            <Link to={loginPath('citizen')}>Đã có tài khoản? Đăng nhập</Link>
            <Link to={ROUTES.home}>Tiếp tục hỏi thủ tục (khách)</Link>
          </div>
        </form>
      </main>
    </div>
  )
}
