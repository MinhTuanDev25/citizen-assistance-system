import { Navigate, useLocation } from 'react-router-dom'
import { useCommune } from '../commune/CommuneContext.jsx'
import { ROUTES } from '../routes.js'

/** Gate: must pick a commune before chat / login / admin. */
export function RequireCommune({ children }) {
  const { commune } = useCommune()
  const location = useLocation()

  if (!commune) {
    return (
      <Navigate to={ROUTES.commune} replace state={{ from: location.pathname }} />
    )
  }

  return children
}
