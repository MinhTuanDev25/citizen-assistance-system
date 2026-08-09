import { Link } from 'react-router-dom'
import { mockStore } from '../../data/mockStore.js'

export default function AdminDashboardPage() {
  const docs = mockStore.documents.length
  const drafts = mockStore.drafts.length
  const active = mockStore.procedures.filter((p) => p.status === 'ACTIVE').length

  return (
    <div className="admin-page">
      <header className="admin-page-head">
        <h1>Tổng quan</h1>
        <p>Pipeline tri thức: upload → extract → review → publish → citizen dùng.</p>
      </header>

      <div className="stat-grid">
        <div className="stat-card">
          <span>Tài liệu</span>
          <strong>{docs}</strong>
        </div>
        <div className="stat-card">
          <span>Bản nháp</span>
          <strong>{drafts}</strong>
        </div>
        <div className="stat-card">
          <span>Thủ tục ACTIVE</span>
          <strong>{active}</strong>
        </div>
      </div>

      <section className="panel">
        <h2>Luồng làm việc</h2>
        <ol className="flow-list">
          <li>
            <Link to="/admin/documents">Upload PDF/text</Link> → lưu documents
          </li>
          <li>Extract → sinh procedure draft (AI)</li>
          <li>
            <Link to="/admin/drafts">Review form</Link> + validate schema
          </li>
          <li>
            Publish version → <Link to="/admin/procedures">thủ tục ACTIVE</Link>
          </li>
          <li>Rollback khi cần (kèm lý do audit)</li>
        </ol>
      </section>
    </div>
  )
}
