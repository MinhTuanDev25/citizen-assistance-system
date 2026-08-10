import { useEffect, useState } from 'react'
import { getActiveVersion, listProcedures } from '../../api/catalog.js'
import { useCommune } from '../../commune/CommuneContext.jsx'

export default function ProceduresPage() {
  const { xaId, commune } = useCommune()
  const [domains, setDomains] = useState([])
  const [count, setCount] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [detail, setDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)

  useEffect(() => {
    if (!xaId) return
    let cancelled = false
    ;(async () => {
      setLoading(true)
      setError('')
      try {
        const data = await listProcedures({ xaId })
        if (cancelled) return
        setDomains(data?.domains || [])
        setCount(data?.count || 0)
      } catch (err) {
        if (!cancelled) setError(err.message || 'Không tải được thủ tục')
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [xaId])

  async function openDefinition(proc) {
    setDetailLoading(true)
    setError('')
    try {
      const data = await getActiveVersion(proc.id)
      setDetail(data)
    } catch (err) {
      setError(err.message || 'Không tải được definition')
    } finally {
      setDetailLoading(false)
    }
  }

  return (
    <div className="admin-page">
      <header className="admin-page-head">
        <h1>Thủ tục ACTIVE</h1>
        <p>
          Xã {commune?.name || xaId} ·{' '}
          <code>GET /api/v1/procedures?xa_id=…</code> · xem definition qua
          active-version. Rollback API chưa có — giữ Phase 4.
        </p>
      </header>

      {error ? <p className="form-error">{error}</p> : null}
      {loading ? <p className="muted">Đang tải…</p> : null}

      {!loading && !error ? (
        <p className="muted" style={{ marginBottom: '1rem' }}>
          Tổng {count} thủ tục · {domains.length} domain
        </p>
      ) : null}

      {domains.map((g) => (
        <div className="panel" key={g.domain_id}>
          <h2>
            {g.domain_name}{' '}
            <span className="pill">{g.count}</span>
          </h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Mã</th>
                  <th>Tiêu đề</th>
                  <th>Active</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {(g.procedures || []).map((p) => (
                  <tr key={p.id}>
                    <td>
                      <code>{p.procedure_code}</code>
                    </td>
                    <td>{p.name}</td>
                    <td>
                      <span className="pill ok">{p.active_version}</span>
                    </td>
                    <td>
                      <button
                        type="button"
                        onClick={() => openDefinition(p)}
                        disabled={detailLoading}
                      >
                        Xem definition
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}

      {detail ? (
        <div className="panel">
          <div className="admin-page-head" style={{ marginBottom: '0.75rem' }}>
            <h2 style={{ margin: 0 }}>
              {detail.procedure_name}{' '}
              <code>{detail.procedure_code}</code> @ {detail.version}
            </h2>
            <button type="button" className="linkish dark" onClick={() => setDetail(null)}>
              Đóng
            </button>
          </div>
          <pre className="json-block">
            {JSON.stringify(detail.definition, null, 2)}
          </pre>
        </div>
      ) : null}
    </div>
  )
}
