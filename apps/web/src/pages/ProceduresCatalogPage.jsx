import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { listProcedures } from '../api/catalog.js'
import { CitizenLayout } from '../components/CitizenLayout.jsx'
import { useCommune } from '../commune/CommuneContext.jsx'

export default function ProceduresCatalogPage() {
  const { xaId, commune } = useCommune()
  const navigate = useNavigate()
  const [domains, setDomains] = useState([])
  const [error, setError] = useState('')

  useEffect(() => {
    if (!xaId) return
    let cancelled = false
    ;(async () => {
      try {
        const data = await listProcedures({ xaId, citizen: true })
        if (cancelled) return
        setDomains(data?.domains || [])
        setError('')
      } catch (err) {
        if (!cancelled) setError(err.message || 'Không tải được danh mục')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [xaId])

  return (
    <CitizenLayout>
      <header className="admin-page-head">
        <h1>Thủ tục hành chính</h1>
        <p>Danh mục đang hỗ trợ tại xã {commune?.name || '…'}.</p>
      </header>
      {error ? <p className="form-error">{error}</p> : null}
      <div className="catalog-list">
        {domains.map((g) => (
          <section key={g.domain_id} className="catalog-block">
            <h2>
              {g.domain_name} · {g.count}
            </h2>
            <ul>
              {(g.procedures || []).map((p) => (
                <li key={p.id}>
                  <div className="catalog-row">
                    <div>
                      <strong>{p.name}</strong>
                      <div className="cell-muted">{p.code || 'Hỏi hướng dẫn tại trợ chuyện'}</div>
                    </div>
                    <button
                      type="button"
                      className="btn-primary catalog-ask"
                      onClick={() => navigate('/hoi', { state: { ask: `Tôi muốn hỏi về thủ tục ${p.name}` } })}
                    >
                      Hỏi hướng dẫn
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </CitizenLayout>
  )
}
