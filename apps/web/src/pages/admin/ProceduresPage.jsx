import { useState } from 'react'
import { mockStore } from '../../data/mockStore.js'

export default function ProceduresPage() {
  const [rows, setRows] = useState(() => [...mockStore.procedures])
  const [msg, setMsg] = useState('')

  function rollback(proc) {
    const reason = window.prompt('Lý do rollback (audit)?', 'Sửa nội dung sai')
    if (reason == null) return
    if (!proc.versions.length) {
      setMsg('Chưa có version để rollback.')
      return
    }
    const prev = proc.versions.find((v) => v.status === 'ARCHIVED') || proc.versions[0]
    proc.versions = proc.versions.map((v) => ({
      ...v,
      status: v.version === prev.version ? 'ACTIVE' : 'ARCHIVED',
    }))
    proc.active_version = prev.version
    mockStore.procedures = mockStore.procedures.map((p) =>
      p.id === proc.id ? { ...proc } : p,
    )
    setRows([...mockStore.procedures])
    setMsg(`Rollback ${proc.procedure_code} → ${prev.version}. Lý do: ${reason}`)
  }

  return (
    <div className="admin-page">
      <header className="admin-page-head">
        <h1>Thủ tục đã publish</h1>
        <p>ACTIVE version phục vụ Citizen chat · rollback kèm audit reason.</p>
      </header>

      {msg ? <p className="form-ok">{msg}</p> : null}

      <div className="panel">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Mã</th>
                <th>Tiêu đề</th>
                <th>Active</th>
                <th>Versions</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => (
                <tr key={p.id}>
                  <td>
                    <code>{p.procedure_code}</code>
                  </td>
                  <td>
                    {p.title}
                    <div className="cell-muted">{p.domain}</div>
                  </td>
                  <td>
                    {p.active_version ? (
                      <span className="pill ok">{p.active_version}</span>
                    ) : (
                      <span className="pill">—</span>
                    )}
                  </td>
                  <td>
                    {(p.versions || [])
                      .map((v) => `${v.version}(${v.status})`)
                      .join(', ') || '—'}
                  </td>
                  <td>
                    {p.active_version ? (
                      <button type="button" onClick={() => rollback(p)}>
                        Rollback
                      </button>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
