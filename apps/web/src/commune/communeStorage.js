const STORAGE_KEY = 'cas.selectedCommune'

export function loadCommune() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (!parsed?.id || !parsed?.name) return null
    return { id: parsed.id, name: parsed.name, description: parsed.description || null }
  } catch {
    return null
  }
}

export function saveCommune(commune) {
  if (!commune) {
    localStorage.removeItem(STORAGE_KEY)
    return
  }
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      id: commune.id,
      name: commune.name,
      description: commune.description || null,
    }),
  )
}
