import { useState } from 'react'
import { Link } from 'react-router-dom'
import { mockStore, nextId } from '../../data/mockStore.js'

export default function DocumentsPage() {
  const [rows, setRows] = useState(() => [...mockStore.documents])
  const [fileName, setFileName] = useState('')
  const [domain, setDomain] = useState('ho_tich_chung_thuc')
  const [msg, setMsg] = useState('')

  function onUpload(e) {
    e.preventDefault()
    if (!fileName.trim()) {
      setMsg('Chọn hoặc nhập tên file.')
      return
    }
    const id = nextId('doc')
    const row = {
      id,
      title: fileName.trim(),
      domain,
      processing_status: 'UPLOADED',
      validity_status: 'PENDING',
      source_type: 'upload',
      uploaded_at: new Date().toISOString(),
    }
    mockStore.documents = [row, ...mockStore.documents]
    setRows([...mockStore.documents])
    setFileName('')
    setMsg(`Đã upload giả lập ${id} (chưa gọi MinIO/API).`)
  }

  function extract(doc) {
    const draftId = nextId('draft')
    const draft = {
      id: draftId,
      procedure_code: 'thu_tuc_moi',
      title: `Draft từ ${doc.title}`,
      status: 'DRAFT',
      document_id: doc.id,
      domain: doc.domain,
      definition: {
        procedure_code: 'thu_tuc_moi',
        title: `Draft từ ${doc.title}`,
        required_slots: ['giay_to'],
        slots: {
          giay_to: {
            type: 'string',
            question: 'Anh/chị mang giấy tờ gì?',
          },
        },
        guidance: {
          checklist: ['CCCD'],
          where_to_submit: 'Bộ phận Một cửa — UBND xã Chư Sê',
        },
      },
      updated_at: new Date().toISOString(),
    }
    mockStore.drafts = [draft, ...mockStore.drafts]
    doc.processing_status = 'EXTRACTED'
    mockStore.documents = mockStore.documents.map((d) =>
      d.id === doc.id ? { ...doc } : d,
    )
    setRows([...mockStore.documents])
    setMsg(`Extract xong → ${draftId}. Vào Bản nháp để duyệt.`)
  }

  return (
    <div className="admin-page">
      <header className="admin-page-head">
        <h1>Tài liệu nguồn</h1>
        <p>Upload PDF/text → extract draft procedure.</p>
      </header>

      <form className="panel form-grid" onSubmit={onUpload}>
        <h2>Upload (demo)</h2>
        <label>
          Tên file / tiêu đề
          <input
            value={fileName}
            onChange={(e) => setFileName(e.target.value)}
            placeholder="vd: hd_chung_thuc.pdf"
          />
        </label>
        <label>
          Domain
          <select value={domain} onChange={(e) => setDomain(e.target.value)}>
            <option value="ho_tich_chung_thuc">Hộ tịch & Chứng thực</option>
            <option value="dat_dai_nha_o">Đất đai, Nhà ở</option>
            <option value="bhxh_chinh_sach">BHXH & Chính sách</option>
          </select>
        </label>
        <button type="submit" className="btn-primary">
          Upload
        </button>
        {msg ? <p className="form-ok">{msg}</p> : null}
      </form>

      <div className="panel">
        <h2>Danh sách</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Tiêu đề</th>
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
                    {d.title}
                    <div className="cell-muted">{d.domain}</div>
                  </td>
                  <td>
                    <span className="pill">{d.processing_status}</span>
                  </td>
                  <td className="row-actions">
                    {d.processing_status === 'UPLOADED' ? (
                      <button type="button" onClick={() => extract(d)}>
                        Extract
                      </button>
                    ) : (
                      <Link to="/admin/drafts">Xem drafts</Link>
                    )}
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
