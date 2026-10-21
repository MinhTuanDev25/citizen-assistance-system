import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { listProcedures } from '../api/catalog.js'
import { useCommune } from '../commune/CommuneContext.jsx'
import { matchesQuery } from '../lib/search.js'
import { ROUTES } from '../routes.js'

export default function CitizenSearch() {
  const navigate = useNavigate()
  const { xaId } = useCommune()
  const wrapRef = useRef(null)
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [procedures, setProcedures] = useState([])

  useEffect(() => {
    if (!xaId) return undefined
    let cancelled = false
    listProcedures({ xaId, citizen: true })
      .then((data) => {
        if (cancelled) return
        const items = (data?.domains || []).flatMap((group) => group.procedures || [])
        setProcedures(items)
      })
      .catch(() => {
        if (!cancelled) setProcedures([])
      })
    return () => {
      cancelled = true
    }
  }, [xaId])

  useEffect(() => {
    function onPointer(event) {
      if (wrapRef.current && !wrapRef.current.contains(event.target)) setOpen(false)
    }
    document.addEventListener('mousedown', onPointer)
    return () => document.removeEventListener('mousedown', onPointer)
  }, [])

  const results = useMemo(() => {
    if (!query.trim()) return []
    return procedures
      .filter((p) => matchesQuery(query, p.name, p.code, p.procedure_code))
      .slice(0, 8)
  }, [query, procedures])

  function pick(name) {
    setOpen(false)
    navigate(ROUTES.chat, { state: { ask: `Tôi muốn hỏi về thủ tục ${name}` } })
  }

  return (
    <div className="cas-search-wrap" ref={wrapRef}>
      <input
        className="cas-search"
        type="search"
        value={query}
        placeholder="Tìm thủ tục, điều kiện…"
        aria-label="Tìm thủ tục"
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
            pick(results[0].name)
          }
        }}
      />
      {open && query.trim() ? (
        <ul className="cas-search-menu" role="listbox">
          {results.length === 0 ? (
            <li className="cas-search-empty">Không có thủ tục khớp.</li>
          ) : results.map((p) => (
            <li key={p.id || p.code || p.name}>
              <button type="button" onClick={() => pick(p.name)}>
                <span className="cas-search-kind">Thủ tục</span>
                <strong>{p.name}</strong>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}
