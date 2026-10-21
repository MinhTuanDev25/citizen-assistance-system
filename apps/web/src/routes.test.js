import { describe, expect, it } from 'vitest'
import {
  LEGACY_REDIRECTS,
  ROUTES,
  adminDocumentPath,
  adminDraftPath,
  adminDraftsPath,
  adminProcedurePath,
  isCommuneSelectPath,
  loginPath,
} from './routes.js'

describe('ROUTES', () => {
  it('uses English path segments', () => {
    const values = Object.values(ROUTES)
    expect(values.every((path) => /^\/[a-z0-9/-]*$/.test(path))).toBe(true)
    expect(ROUTES.chat).toBe('/chat')
    expect(ROUTES.procedures).toBe('/procedures')
    expect(ROUTES.commune).toBe('/commune')
  })

  it('redirects legacy Vietnamese slugs', () => {
    const map = Object.fromEntries(LEGACY_REDIRECTS.map((row) => [row.from, row.to]))
    expect(map['/hoi']).toBe('/chat')
    expect(map['/thu-tuc']).toBe('/procedures')
    expect(map['/chon-xa']).toBe('/commune')
    expect(map['/giong-noi']).toBe('/chat')
    expect(map['/ho-so']).toBe('/')
  })

  it('treats commune select aliases as the same gate', () => {
    expect(isCommuneSelectPath('/commune')).toBe(true)
    expect(isCommuneSelectPath('/chon-xa')).toBe(true)
    expect(isCommuneSelectPath('/chat')).toBe(false)
  })

  it('builds query helpers without Vietnamese slugs', () => {
    expect(loginPath('citizen')).toBe('/login?as=citizen')
    expect(adminDocumentPath('abc')).toBe('/admin/documents?doc=abc')
    expect(adminProcedurePath('p1')).toBe('/admin/procedures?proc=p1')
    expect(adminDraftsPath('p1')).toBe('/admin/drafts?proc=p1')
    expect(adminDraftPath('d1')).toBe('/admin/drafts/d1')
  })
})
