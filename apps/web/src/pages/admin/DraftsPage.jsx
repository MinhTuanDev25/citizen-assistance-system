import { Link } from 'react-router-dom'
import { mockStore } from '../../data/mockStore.js'

export default function DraftsPage() {
  const rows = mockStore.drafts

  return (
    <div className="admin-page">
      <header className="admin-page-head">
        <h1>Bản nháp thủ tục</h1>
        <p>Review form (không sửa raw JSON) → validate → publish.</p>
      </header>

      <div className="panel">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Mã / tiêu đề</th>
                <th>Status</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((d) => (
                <tr key={d.id}>
                  <td>
                    <code>{d.id}</code>
                  </td>
                  <td>
                    <strong>{d.procedure_code}</strong>
                    <div className="cell-muted">{d.title}</div>
                  </td>
                  <td>
                    <span className="pill">{d.status}</span>
                  </td>
                  <td>
                    <Link to={`/admin/drafts/${d.id}`}>Review</Link>
                  </td>
                </tr>
              ))}
              {rows.length === 0 ? (
                <tr>
                  <td colSpan={4}>Chưa có draft — upload & extract trước.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
