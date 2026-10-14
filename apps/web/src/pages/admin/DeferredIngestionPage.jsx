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
        </p>
      </header>
      <div className="panel">
        <p className="muted">
          Không có dữ liệu giả. Upload PDF và kích hoạt phiên bản nằm ở Tài liệu và Thủ tục.
        </p>
      </div>
    </div>
  )
}
