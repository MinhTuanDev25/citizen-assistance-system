import { Navigate, useLocation } from 'react-router-dom'
import { useCommune } from '../commune/CommuneContext.jsx'

/** Gate: must pick a commune before chat / login / admin. */
export function RequireCommune({ children }) {
  const { commune } = useCommune()
  const location = useLocation()

  if (!commune) {
    return (
      <Navigate to="/chon-xa" replace state={{ from: location.pathname }} />
    )
  }

  return children
}
