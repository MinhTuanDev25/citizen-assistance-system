/**
 * Build-time feature flags (Vite replaces import.meta.env.*).
 * Production CI sets VITE_ADMIN_INGESTION=false so the real upload
 * screen is tree-shaken out of the bundle. Draft/OCR stays unimplemented.
 */
export function adminIngestionEnabled() {
  const v = import.meta.env.VITE_ADMIN_INGESTION
  return v === 'true' || v === '1'
}

export function adminIndexingEnabled() {
  const v = import.meta.env.VITE_ADMIN_INDEXING
  return v === 'true' || v === '1'
}
