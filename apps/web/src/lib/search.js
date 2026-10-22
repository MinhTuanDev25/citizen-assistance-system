export function foldQuery(value) {
  return String(value || '')
    .toLowerCase()
    .normalize('NFD')
    .replace(/\p{M}/gu, '')
    .trim()
}

export function matchesQuery(query, ...parts) {
  const needle = foldQuery(query)
  if (!needle) return false
  return parts.some((part) => foldQuery(part).includes(needle))
}
