import { createContext, useContext, useMemo, useState } from 'react'
import { loadCommune, saveCommune } from './communeStorage.js'

const CommuneContext = createContext(null)

export function CommuneProvider({ children }) {
  const [commune, setCommuneState] = useState(() => loadCommune())

  const value = useMemo(
    () => ({
      commune,
      xaId: commune?.id || null,
      setCommune(next) {
        saveCommune(next)
        setCommuneState(next)
      },
      clearCommune() {
        saveCommune(null)
        setCommuneState(null)
      },
    }),
    [commune],
  )

  return (
    <CommuneContext.Provider value={value}>{children}</CommuneContext.Provider>
  )
}

export function useCommune() {
  const ctx = useContext(CommuneContext)
  if (!ctx) throw new Error('useCommune outside CommuneProvider')
  return ctx
}
