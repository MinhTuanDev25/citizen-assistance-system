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
          Nhận dạng chữ, tạo bản nháp và phát hành <strong>chưa được triển khai</strong>.
        </p>
      </header>
      <div className="panel">
        <p className="muted">
          Không có dữ liệu minh họa. Tải tài liệu lên và kích hoạt phiên bản tại mục Tài liệu và Thủ tục.
        </p>
      </div>
    </div>
  )
}
