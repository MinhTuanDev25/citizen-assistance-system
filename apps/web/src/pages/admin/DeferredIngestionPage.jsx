/**
 * Placeholder when PDF → AI draft → review → publish is not implemented.
 * Keeps admin routes navigable without fake production data.
 */
export default function DeferredIngestionPage({ title }) {
  return (
    <div className="admin-page">
      <header className="admin-page-head">
        <h1>{title || 'Pipeline tri thức'}</h1>
        <p>
          OCR, extract bản nháp và publish <strong>chưa triển khai</strong>.
          Upload PDF thật chỉ có khi build với <code>VITE_ADMIN_INGESTION=true</code>.
        </p>
      </header>
      <div className="panel">
        <p className="cell-muted">
          Bản build production mặc định tắt ingestion nên không nhúng dữ liệu giả.
        </p>
      </div>
    </div>
  )
}
