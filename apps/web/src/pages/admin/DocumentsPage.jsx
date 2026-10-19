import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { listDomains } from '../../api/catalog.js'
import {
  downloadDocument,
  linkDocument,
  listDocumentLinks,
  listDocuments,
  listLinkTargets,
  requestIndexing,
  reindexSettledReady,
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

function pillClass(status) {
  const u = String(status || '').toUpperCase()
  if (['READY', 'ACTIVE', 'PROCESSED', 'VALID'].includes(u)) return 'pill ok'
  if (['FAILED', 'EXPIRED', 'SUPERSEDED'].includes(u)) return 'pill bad'
  if (['PROCESSING', 'UPLOADED', 'APPROVED', 'INDEXING'].includes(u)) return 'pill warn'
  if (['PENDING', 'ARCHIVED'].includes(u)) return 'pill mute'
  return 'pill'
}

const PROCESS_LABEL = {
  UPLOADED: 'Đã tải',
  PROCESSING: 'Đang xử lý',
  PROCESSED: 'Đã xử lý',
  READY: 'Sẵn sàng',
  FAILED: 'Lỗi',
}

function statusLabel(map, status) {
  const code = String(status || '').toUpperCase()
  return map[code] || status || '—'
}

function displayFilename(name) {
  if (!name) return ''
  try {
    return decodeURIComponent(String(name).replace(/\+/g, ' '))
  } catch {
    return name
  }
}

function linkHeading(link, targets) {
  const target = targets.find((item) => item.procedure_version_id === link.procedure_version_id)
  const name = target?.procedure_name || link.procedure_name || link.procedure_code
  return `${name} · ${link.version}`
}

function linkNote(link, targets) {
  const status = String(link.index_status || '').toUpperCase()
  if (status === 'FAILED') {
    if (link.last_error_code === 'page_limit') {
      return 'PDF vượt giới hạn số trang khi lập chỉ mục. Thử lại sau khi tăng hạn mức, hoặc dùng bản rút gọn.'
    }
    if (link.last_error_code === 'timeout') {
      return 'Lập chỉ mục quá thời gian (PDF nhiều trang). Bấm Thử lại một lần và đợi — có thể mất tới 15 phút.'
    }
    return 'Lập chỉ mục thất bại. Thử lại hoặc bỏ liên kết.'
  }
  if (status !== 'READY') return ''
  const target = targets.find((item) => item.procedure_version_id === link.procedure_version_id)
  if (target?.status && target.status !== 'ACTIVE') {
    return `Phiên ${link.version} chưa đang dùng. Công dân vẫn hỏi phiên đang kích hoạt.`
  }
  const pages = link.page_count
  const chunks = link.chunk_count
  if (pages == null && chunks == null) return ''
  return `${pages ?? 0} trang · ${chunks ?? 0} đoạn`
}

export default function DocumentsPage() {
  const [params] = useSearchParams()
  const focusDoc = params.get('doc')
  const focusApplied = useRef('')
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
  const detailRef = useRef(null)
  const [targets, setTargets] = useState([])
  const [links, setLinks] = useState([])
  const [versionId, setVersionId] = useState('')
  const [pageRange, setPageRange] = useState('')
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

  function openDetail(doc) {
    if (selected?.id === doc.id) {
      const node = detailRef.current
      if (node && typeof node.scrollIntoView === 'function') {
        node.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
      return
    }
    setSelected(doc)
  }

  useEffect(() => {
    if (!focusDoc || focusApplied.current === focusDoc) return undefined
    let cancelled = false
    listDocuments({ limit: 100 })
      .then((data) => {
        if (cancelled) return
        const found = (data?.items || []).find((d) => d.id === focusDoc)
        if (!found) return
        focusApplied.current = focusDoc
        if (found.domain_id) setDomainId(found.domain_id)
        openDetail(found)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [focusDoc])

  useLayoutEffect(() => {
    if (!selected) return
    const node = detailRef.current
    if (!node || typeof node.scrollIntoView !== 'function') return
    node.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [selected])

  useEffect(() => {
    if (!indexingOn || !selected?.id) {
      setLinks([])
      return undefined
    }
    const documentId = selected.id
    let cancel = false
    listDocumentLinks(documentId)
      .then((data) => {
        if (!cancel) setLinks(data.items || [])
      })
      .catch((err) => {
        if (!cancel) setError(err.message || 'Không tải được liên kết.')
      })
    return () => {
      cancel = true
    }
  }, [selected?.id])

  async function onLink(event) {
    event.preventDefault()
    if (!selected || !versionId) return
    setError('')
    setIndexMsg('')
    try {
      await linkDocument(selected.id, {
        procedure_version_id: versionId,
        relationship_type: 'SOURCE',
        page_range: pageRange.trim(),
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
      const result = await requestReindex(selected.id, versionId, pageRange)
      setSelected((prev) => (prev ? { ...prev, processing_status: result.processing_status } : prev))
      setIndexMsg(reindexSettledReady(result) ? 'Đã lập lại chỉ mục.' : 'Lập lại chỉ mục chưa thay thế bản đang dùng.')
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
        <div>
          <h1>Tài liệu</h1>
          <p>PDF đã gắn và Sẵn sàng được trích dẫn khi công dân hỏi phiên đang dùng. Bỏ liên kết thì lượt sau không còn đoạn từ file đó.</p>
        </div>
      </header>

      <form className="panel upload-grid" onSubmit={onUpload}>
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
        <div className="upload-submit">
          <button type="submit" className="btn-primary" disabled={busy}>
            {busy ? 'Đang tải lên…' : 'Upload'}
          </button>
        </div>
        {(file || error || msg) ? (
          <div className="upload-actions">
            {file ? (
              <span className="cell-muted">
                Đã chọn {displayFilename(file.name)} ({formatSize(file.size)})
              </span>
            ) : null}
            {error ? <p className="form-error">{error}</p> : null}
            {msg ? <p className="form-ok">{msg}</p> : null}
          </div>
        ) : null}
      </form>

      <div className="panel">
        <div className="panel-head">
          <h2>Danh sách</h2>
          <label className="inline-filter">
            Trạng thái xử lý
            <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
              <option value="">Tất cả</option>
              <option value="UPLOADED">Đã tải</option>
              <option value="PROCESSING">Đang xử lý</option>
              <option value="PROCESSED">Đã xử lý</option>
              <option value="READY">Sẵn sàng</option>
              <option value="FAILED">Lỗi</option>
            </select>
          </label>
        </div>
        <div className="table-wrap">
          <table className="doc-table">
            <thead>
              <tr>
                <th>Tiêu đề</th>
                <th className="col-status">Xử lý</th>
                <th className="col-actions" />
              </tr>
            </thead>
            <tbody>
              {rows.map((d) => (
                <tr key={d.id} className={selected?.id === d.id ? 'is-selected' : ''}>
                  <td>
                    <strong>{d.title}</strong>
                    <div className="cell-muted file-name" title={displayFilename(d.filename)}>
                      {displayFilename(d.filename)}
                    </div>
                  </td>
                  <td className="col-status">
                    <span className={pillClass(d.processing_status)}>
                      {statusLabel(PROCESS_LABEL, d.processing_status)}
                    </span>
                  </td>
                  <td className="col-actions">
                    <div className="row-actions">
                      <button type="button" onClick={() => openDetail(d)}>
                        Chi tiết
                      </button>
                      <button type="button" onClick={() => { void onDownload(d) }}>
                        Tải PDF
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {selected ? (
        <div className="panel anchor-target" ref={detailRef}>
          <div className="panel-head">
            <h2>{selected.title}</h2>
            <button type="button" className="linkish dark" onClick={() => setSelected(null)}>
              Đóng
            </button>
          </div>
          <div className="meta-row">
            <span className={pillClass(selected.processing_status)}>
              {statusLabel(PROCESS_LABEL, selected.processing_status)}
            </span>
            <span className="cell-muted">{formatSize(selected.file_size_bytes || 0)}</span>
          </div>
          <p className="cell-muted file-name">{displayFilename(selected.filename)}</p>
          {indexingOn ? (
            <form className="link-form" onSubmit={onLink}>
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
              <label>
                Phạm vi trang
                <input
                  name="page_range"
                  value={pageRange}
                  onChange={(e) => setPageRange(e.target.value)}
                  placeholder="10-21 · trống = giữ khi lập lại · all = toàn bộ"
                />
              </label>
              <button type="submit" className="btn-primary">Gắn thủ tục</button>
              <ul className="link-list">
                {links.map((link) => {
                  const note = linkNote(link, targets)
                  return (
                  <li key={link.procedure_version_id}>
                    <div>
                      <strong>{linkHeading(link, targets)}</strong>
                      <div className="cell-muted">
                        <span className={pillClass(link.index_status)}>
                          {statusLabel(PROCESS_LABEL, link.index_status)}
                        </span>
                        {` · trang ${link.page_range || 'toàn bộ'}`}
                        {note ? ` · ${note}` : ''}
                      </div>
                    </div>
                    <div className="row-actions">
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
                    </div>
                  </li>
                  )
                })}
              </ul>
              {indexMsg ? <p className="form-ok">{indexMsg}</p> : null}
            </form>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
