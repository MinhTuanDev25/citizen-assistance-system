const STORAGE_KEY = 'cas_auth'

export function loadAuth() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export function saveAuth(user) {
  if (!user) {
    localStorage.removeItem(STORAGE_KEY)
    return
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(user))
}

/** Demo accounts — replace with JWT later */
export function login(email, password) {
  const e = email.trim().toLowerCase()
  if (e === 'admin@chuse.vn' && password === 'admin123') {
    return {
      id: 'u_admin',
      email: 'admin@chuse.vn',
      name: 'Cán bộ One Cửa',
      role: 'ADMIN',
    }
  }
  if (e === 'citizen@example.com' && password === 'citizen123') {
    return {
      id: 'u_citizen',
      email: 'citizen@example.com',
      name: 'Công dân demo',
      role: 'CITIZEN',
    }
  }
  throw new Error('Email hoặc mật khẩu không đúng')
}
