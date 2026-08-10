const STORAGE_KEY = 'cas_auth'

export function loadAuth() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (!parsed?.id || !parsed?.role || !parsed?.accessToken) return null
    return {
      id: parsed.id,
      email: parsed.email || '',
      name: parsed.name || '',
      role: parsed.role,
      accessToken: parsed.accessToken,
    }
  } catch {
    return null
  }
}

export function saveAuth(user) {
  if (!user) {
    localStorage.removeItem(STORAGE_KEY)
    return
  }
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      id: user.id,
      email: user.email,
      name: user.name,
      role: user.role,
      accessToken: user.accessToken,
    }),
  )
}
