import { createContext, useContext, useMemo, useState } from 'react'
import { loginRequest, logoutRequest } from '../api/auth.js'
import { clearChatSession } from '../api/session.js'
import { loadAuth, saveAuth } from './auth.js'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => loadAuth())

  const value = useMemo(
    () => ({
      user,
      accessToken: user?.accessToken || null,
      isAdmin: user?.role === 'ADMIN',
      async login(email, password) {
        const next = await loginRequest(email, password)
        saveAuth(next)
        setUser(next)
        // New auth identity → fresh chat session next time
        clearChatSession()
        return next
      },
      async logout() {
        await logoutRequest()
        saveAuth(null)
        setUser(null)
        clearChatSession()
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
