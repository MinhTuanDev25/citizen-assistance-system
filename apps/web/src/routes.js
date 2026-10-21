/** Canonical frontend paths. UI copy stays Vietnamese; URLs are English. */

export const ROUTES = Object.freeze({
  home: '/',
  chat: '/chat',
  procedures: '/procedures',
  commune: '/commune',
  login: '/login',
  admin: '/admin',
  adminDocuments: '/admin/documents',
  adminProcedures: '/admin/procedures',
  adminDrafts: '/admin/drafts',
})

/** Old Vietnamese slugs kept as redirects so bookmarks still work. */
export const LEGACY_REDIRECTS = Object.freeze([
  { from: '/hoi', to: ROUTES.chat },
  { from: '/thu-tuc', to: ROUTES.procedures },
  { from: '/chon-xa', to: ROUTES.commune },
  { from: '/ho-so', to: ROUTES.home },
  { from: '/giong-noi', to: ROUTES.chat },
])

export function isCommuneSelectPath(pathname) {
  return pathname === ROUTES.commune || pathname === '/chon-xa'
}

export function loginPath(as) {
  if (!as) return ROUTES.login
  return `${ROUTES.login}?as=${encodeURIComponent(as)}`
}

export function adminDocumentPath(id) {
  return `${ROUTES.adminDocuments}?doc=${encodeURIComponent(id)}`
}

export function adminProcedurePath(id) {
  return `${ROUTES.adminProcedures}?proc=${encodeURIComponent(id)}`
}

export function adminDraftsPath(procedureId) {
  if (!procedureId) return ROUTES.adminDrafts
  return `${ROUTES.adminDrafts}?proc=${encodeURIComponent(procedureId)}`
}

export function adminDraftPath(id) {
  return `${ROUTES.adminDrafts}/${encodeURIComponent(id)}`
}
