import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { listAdminProcedures } from '../api/catalog.js'
import { listDocuments } from '../api/documents.js'
import { foldQuery, matchesQuery } from '../lib/search.js'
import { adminDocumentPath, adminProcedurePath } from '../routes.js'

const LIMIT = 8

export default function AdminSearch() {
  const navigate = useNavigate()
  const wrapRef = useRef(null)
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [docs, setDocs] = useState([])
  const [procedures, setProcedures] = useState([])

  useEffect(() => {
    let cancelled = false
    Promise.all([
      listDocuments({ limit: 100 }).catch(() => ({ items: [] })),
      listAdminProcedures().catch(() => ({ items: [] })),
    ]).then(([docData, procData]) => {
      if (cancelled) return
      setDocs(docData?.items || [])
      setProcedures(procData?.items || [])
    })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    function onPointer(event) {
      if (wrapRef.current && !wrapRef.current.contains(event.target)) setOpen(false)
    }
    document.addEventListener('mousedown', onPointer)
    return () => document.removeEventListener('mousedown', onPointer)
  }, [])

  const results = useMemo(() => {
    if (!foldQuery(query)) return []
    const needle = query
    const docHits = docs
      .filter((d) => matchesQuery(needle, d.title, d.filename, d.document_number, d.issuer))
      .map((d) => ({
        key: `doc-${d.id}`,
        kind: 'Tài liệu',
        label: d.title || d.filename || 'Tài liệu',
        hint: d.filename || '',
        href: adminDocumentPath(d.id),
      }))
    const procHits = procedures
      .filter((p) => matchesQuery(needle, p.name, p.procedure_code, p.active_version))
      .map((p) => ({
        key: `proc-${p.id}`,
        kind: 'Thủ tục',
        label: p.name || p.procedure_code,
        hint: p.active_version ? `Đang dùng ${p.active_version}` : 'Chưa có phiên đang dùng',
        href: adminProcedurePath(p.id),
      }))
    return [...docHits, ...procHits].slice(0, LIMIT)
  }, [query, docs, procedures])

  function pick(href) {
    setOpen(false)
    navigate(href)
  }

  return (
    <div className="cas-search-wrap" ref={wrapRef}>
      <input
        className="cas-search"
        type="search"
        value={query}
        placeholder="Tìm tài liệu, thủ tục…"
        aria-label="Tìm tài liệu, thủ tục"
        autoComplete="off"
        onChange={(e) => {
          setQuery(e.target.value)
          setOpen(true)
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e) => {
          if (e.key === 'Escape') setOpen(false)
          if (e.key === 'Enter' && results[0]) {
            e.preventDefault()
            pick(results[0].href)
          }
        }}
      />
      {open && query.trim() ? (
        <ul className="cas-search-menu" role="listbox">
          {results.length === 0 ? (
            <li className="cas-search-empty">Không có kết quả.</li>
          ) : results.map((hit) => (
            <li key={hit.key}>
              <button type="button" onClick={() => pick(hit.href)}>
                <span className="cas-search-kind">{hit.kind}</span>
                <strong>{hit.label}</strong>
                {hit.hint ? <span className="cell-muted">{hit.hint}</span> : null}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}
