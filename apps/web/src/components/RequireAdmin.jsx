import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext.jsx'
import { ROUTES } from '../routes.js'

export function RequireAdmin({ children }) {
  const { user, isAdmin } = useAuth()
  const location = useLocation()

  if (!user) {
    return <Navigate to={ROUTES.login} replace state={{ from: location.pathname }} />
  }
  if (!isAdmin) {
    return <Navigate to={ROUTES.home} replace />
  }
  return children
}
