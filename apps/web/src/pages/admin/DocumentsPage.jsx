import { useEffect, useRef, useState } from 'react'
import { listDomains } from '../../api/catalog.js'
import {
  downloadDocument,
  linkDocument,
  listDocumentLinks,
  listDocuments,
  listLinkTargets,
  requestIndexing,
  requestReindex,
  unlinkDocument,
  uploadDocument,
  uploadErrorMessage,
  validateUpload,
} from '../../api/documents.js'
import { adminIndexingEnabled } from '../../config/features.js'

const indexingOn = adminIndexingEnabled()

function formatSize(n) {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

export default function DocumentsPage() {
  const [rows, setRows] = useState([])
  const [domains, setDomains] = useState([])
  const [file, setFile] = useState(null)
  const [title, setTitle] = useState('')
  const [domainId, setDomainId] = useState('')
  const [documentNumber, setDocumentNumber] = useState('')
  const [issuer, setIssuer] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [busy, setBusy] = useState(false)
  const busyRef = useRef(false)
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')
  const [selected, setSelected] = useState(null)
  const [targets, setTargets] = useState([])
  const [links, setLinks] = useState([])
  const [versionId, setVersionId] = useState('')
  const [indexMsg, setIndexMsg] = useState('')
  const [maxBytes, setMaxBytes] = useState(0)
  const [busyVersionIds, setBusyVersionIds] = useState([])
  const linkBusyRef = useRef(new Set())

  async function refresh(filter = statusFilter) {
    const data = await listDocuments({
      domain_id: domainId,
      processing_status: filter,
    })
    setRows(data?.items || [])
    if (typeof data?.max_bytes === 'number' && data.max_bytes > 0) setMaxBytes(data.max_bytes)
  }

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const data = await listDomains({ active: true })
        if (cancelled) return
        const items = data?.items || []
        setDomains(items)
        if (items[0]) setDomainId((prev) => prev || items[0].id)
      } catch (err) {
        if (!cancelled) setError(uploadErrorMessage(err.status || 500))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!domainId) return
    let cancelled = false
    ;(async () => {
      try {
        const data = await listDocuments({ domain_id: domainId, processing_status: statusFilter })
        if (!cancelled) {
          setRows(data?.items || [])
          if (typeof data?.max_bytes === 'number' && data.max_bytes > 0) setMaxBytes(data.max_bytes)
        }
      } catch (err) {
        if (!cancelled) setError(uploadErrorMessage(err.status || 500))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [domainId, statusFilter])

  async function onUpload(e) {
    e.preventDefault()
    if (busyRef.current) return
    const problem = validateUpload({ file, title, domainId, maxBytes })
    if (problem) {
      setError(problem)
      setMsg('')
      return
    }
    busyRef.current = true
    setBusy(true)
    setError('')
    setMsg('')
    try {
      await uploadDocument({ file, title, domainId, documentNumber, issuer })
      setMsg('Đã lưu PDF.')
      setFile(null)
      setTitle('')
      setDocumentNumber('')
      setIssuer('')
      await refresh()
    } catch (err) {
      setError(err.message || uploadErrorMessage(err.status || 500))
    } finally {
      busyRef.current = false
      setBusy(false)
    }
  }

  useEffect(() => {
    if (!indexingOn) return undefined
    let cancel = false
    listLinkTargets()
      .then((data) => {
        if (!cancel) setTargets(data.items || [])
      })
      .catch((err) => {
        if (!cancel) setError(err.message || 'Không tải được thủ tục.')
      })
    return () => {
      cancel = true
    }
  }, [])

  useEffect(() => {
    if (!indexingOn || !selected?.id) {
      setLinks([])
      return undefined
    }
    let cancel = false
    listDocumentLinks(selected.id)
      .then((data) => {
        if (!cancel) setLinks(data.items || [])
      })
      .catch((err) => {
        if (!cancel) setError(err.message || 'Không tải được liên kết.')
      })
    return () => {
      cancel = true
    }
  }, [selected])

  async function onLink(event) {
    event.preventDefault()
    if (!selected || !versionId) return
    setError('')
    setIndexMsg('')
    try {
      await linkDocument(selected.id, {
        procedure_version_id: versionId,
        relationship_type: 'SOURCE',
        page_range: '',
      })
      const data = await listDocumentLinks(selected.id)
      setLinks(data.items || [])
      setIndexMsg('Đã gắn thủ tục.')
    } catch (err) {
      setError(err.message || 'Không gắn được thủ tục.')
    }
  }

  async function reloadLinks(id) {
    const data = await listDocumentLinks(id)
    setLinks(data.items || [])
  }

  function setLinkBusy(versionId, on) {
    const next = new Set(linkBusyRef.current)
    if (on) next.add(versionId)
    else next.delete(versionId)
    linkBusyRef.current = next
    setBusyVersionIds([...next])
  }

  async function onReindex(versionId) {
    if (!selected || linkBusyRef.current.has(versionId)) return
    setLinkBusy(versionId, true)
    setError('')
    setIndexMsg('')
    try {
      const result = await requestReindex(selected.id, versionId)
      setSelected((prev) => (prev ? { ...prev, processing_status: result.processing_status } : prev))
      setIndexMsg(result.link_status === 'READY' && !result.error_code ? 'Đã lập lại chỉ mục.' : 'Lập lại chỉ mục chưa thay thế bản đang dùng.')
      await reloadLinks(selected.id)
      await refresh()
    } catch (err) {
      setError(err.message || 'Không lập lại chỉ mục được.')
    } finally {
      setLinkBusy(versionId, false)
    }
  }

  async function onIndex(versionId, retry) {
    if (!selected || linkBusyRef.current.has(versionId)) return
    setLinkBusy(versionId, true)
    setError('')
    setIndexMsg('')
    try {
      const result = await requestIndexing(selected.id, versionId, retry)
      setSelected((prev) => (prev ? { ...prev, processing_status: result.processing_status } : prev))
      setIndexMsg(result.link_status === 'READY' ? 'Đã lập chỉ mục.' : 'Lập chỉ mục thất bại.')
      await reloadLinks(selected.id)
      await refresh()
    } catch (err) {
      setError(err.message || 'Không lập chỉ mục được.')
    } finally {
      setLinkBusy(versionId, false)
    }
  }

  async function onUnlink(versionId) {
    if (!selected || linkBusyRef.current.has(versionId)) return
    setLinkBusy(versionId, true)
    setError('')
    setIndexMsg('')
    try {
      await unlinkDocument(selected.id, versionId)
      await reloadLinks(selected.id)
      await refresh()
      setIndexMsg('Đã bỏ liên kết.')
    } catch (err) {
      setError(err.message || 'Không bỏ liên kết được.')
    } finally {
      setLinkBusy(versionId, false)
    }
  }

  async function onDownload(doc) {
    setError('')
    try {
      await downloadDocument(doc.id, doc.filename)
    } catch (err) {
      setError(err.message || uploadErrorMessage(err.status || 500))
    }
  }

  return (
    <div className="admin-page">
      <header className="admin-page-head">
        <h1>Tài liệu nguồn</h1>
        <p>Upload PDF thật. OCR và bản nháp thủ tục chưa triển khai.</p>
      </header>

      <form className="panel form-grid" onSubmit={onUpload}>
        <h2>Upload PDF</h2>
        <label>
          File PDF
          <input
            name="file"
            type="file"
            accept="application/pdf,.pdf"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
        </label>
        {file ? (
          <p className="cell-muted">
            Đã chọn {file.name} ({formatSize(file.size)})
          </p>
        ) : null}
        <label>
          Tiêu đề
          <input name="title" value={title} onChange={(e) => setTitle(e.target.value)} />
        </label>
        <label>
          Lĩnh vực
          <select name="domain" value={domainId} onChange={(e) => setDomainId(e.target.value)}>
            {domains.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Số hiệu
          <input name="document_number" value={documentNumber} onChange={(e) => setDocumentNumber(e.target.value)} />
        </label>
        <label>
          Cơ quan ban hành
          <input name="issuer" value={issuer} onChange={(e) => setIssuer(e.target.value)} />
        </label>
        <button type="submit" className="btn-primary" disabled={busy}>
          {busy ? 'Đang tải lên…' : 'Upload'}
        </button>
        {error ? <p className="form-error">{error}</p> : null}
        {msg ? <p className="form-ok">{msg}</p> : null}
      </form>

      <div className="panel">
        <h2>Danh sách</h2>
        <label>
          Trạng thái xử lý
          <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="">Tất cả</option>
            <option value="UPLOADED">UPLOADED</option>
            <option value="PROCESSING">PROCESSING</option>
            <option value="PROCESSED">PROCESSED</option>
            <option value="READY">READY</option>
            <option value="FAILED">FAILED</option>
          </select>
        </label>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Tiêu đề</th>
                <th>Xử lý</th>
                <th>Hiệu lực</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((d) => (
                <tr key={d.id}>
                  <td>
                    {d.title}
                    <div className="cell-muted">{d.filename}</div>
                  </td>
                  <td>
                    <span className="pill">{d.processing_status}</span>
                  </td>
                  <td>
                    <span className="pill">{d.validity_status}</span>
                  </td>
                  <td className="row-actions">
                    <button type="button" onClick={() => setSelected(d)}>
                      Chi tiết
                    </button>
                    <button type="button" onClick={() => { void onDownload(d) }}>
                      Tải PDF
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {selected ? (
          <div className="panel">
            <h3>{selected.title}</h3>
            <p>Xử lý: {selected.processing_status}</p>
            <p>Hiệu lực: {selected.validity_status}</p>
            <p>Checksum: {selected.checksum}</p>
            <p>Dung lượng: {formatSize(selected.file_size_bytes || 0)}</p>
            {indexingOn ? (
              <form onSubmit={onLink}>
                <p>Trạng thái lập chỉ mục: {selected.processing_status}</p>
                <label>
                  Thủ tục / phiên bản
                  <select name="procedure_version_id" value={versionId} onChange={(e) => setVersionId(e.target.value)}>
                    <option value="">Chọn phiên bản</option>
                    {targets.map((item) => (
                      <option key={item.procedure_version_id} value={item.procedure_version_id}>
                        {item.procedure_name} · {item.version}
                      </option>
                    ))}
                  </select>
                </label>
                <button type="submit">Gắn thủ tục</button>
                <ul>
                  {links.map((link) => (
                    <li key={link.procedure_version_id}>
                      {link.procedure_code} · {link.version} · {link.index_status}
                      {link.last_error_code ? ` · ${link.last_error_code}` : ''}
                      {link.reindex_error_code ? ` · reindex ${link.reindex_error_code}` : ''}
                      {link.pipeline_version ? ` · ${link.page_count ?? 0} trang · ${link.chunk_count ?? 0} đoạn · OCR ${link.ocr_page_count ?? 0} · ${link.pipeline_version}${link.embedding_model_id ? ` · ${link.embedding_model_id}` : ''}` : ''}
                      {link.index_status === 'UPLOADED' ? (
                        <button type="button" disabled={busyVersionIds.includes(link.procedure_version_id)} onClick={() => { void onIndex(link.procedure_version_id, false) }}>
                          Lập chỉ mục
                        </button>
                      ) : null}
                      {link.index_status === 'READY' ? (
                        <button type="button" disabled={busyVersionIds.includes(link.procedure_version_id)} onClick={() => { void onReindex(link.procedure_version_id) }}>
                          Lập lại chỉ mục
                        </button>
                      ) : null}
                      {link.index_status === 'FAILED' || link.recoverable ? (
                        <button type="button" disabled={busyVersionIds.includes(link.procedure_version_id)} onClick={() => { void onIndex(link.procedure_version_id, true) }}>
                          Thử lại lập chỉ mục
                        </button>
                      ) : null}
                      {link.index_status !== 'PROCESSING' ? (
                        <button type="button" disabled={busyVersionIds.includes(link.procedure_version_id)} onClick={() => { void onUnlink(link.procedure_version_id) }}>
                          Bỏ liên kết
                        </button>
                      ) : null}
                    </li>
                  ))}
                </ul>
                {indexMsg ? <p className="form-ok">{indexMsg}</p> : null}
              </form>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  )
}
