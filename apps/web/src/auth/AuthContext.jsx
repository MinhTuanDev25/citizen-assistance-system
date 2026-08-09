import { createContext, useContext, useMemo, useState } from 'react'
import { loadAuth, login as doLogin, saveAuth } from './auth.js'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => loadAuth())

  const value = useMemo(
    () => ({
      user,
      isAdmin: user?.role === 'ADMIN',
      login(email, password) {
        const next = doLogin(email, password)
        saveAuth(next)
        setUser(next)
        return next
      },
      logout() {
        saveAuth(null)
        setUser(null)
      },
    }),
    [user],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth outside AuthProvider')
  return ctx
}
