import { useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { mockStore } from '../../data/mockStore.js'

export default function DraftReviewPage() {
  const { draftId } = useParams()
  const navigate = useNavigate()
  const initial = useMemo(
    () => mockStore.drafts.find((d) => d.id === draftId),
    [draftId],
  )

  const [draft, setDraft] = useState(() =>
    initial ? structuredClone(initial) : null,
  )
  const [errors, setErrors] = useState([])
  const [ok, setOk] = useState('')

  if (!draft) {
    return (
      <div className="admin-page">
        <p>Không tìm thấy draft.</p>
        <Link to="/admin/drafts">← Danh sách</Link>
      </div>
    )
  }

  const def = draft.definition

  function updateSlot(key, field, value) {
    setDraft((prev) => {
      const next = structuredClone(prev)
      next.definition.slots[key][field] = value
      return next
    })
    setOk('')
  }

  function updateGuidance(field, value) {
    setDraft((prev) => {
      const next = structuredClone(prev)
      if (field === 'checklist') {
        next.definition.guidance.checklist = value.split('\n').filter(Boolean)
      } else {
        next.definition.guidance[field] = value
      }
      return next
    })
    setOk('')
  }

  function validate() {
    const errs = []
    if (!def.procedure_code?.trim()) errs.push({ path: 'procedure_code', message: 'Thiếu mã thủ tục' })
    if (!def.title?.trim()) errs.push({ path: 'title', message: 'Thiếu tiêu đề' })
    for (const key of def.required_slots || []) {
      if (!def.slots[key]) {
        errs.push({ path: `required_slots.${key}`, message: 'Slot required chưa định nghĩa' })
      } else if (!def.slots[key].question?.trim()) {
        errs.push({ path: `slots.${key}.question`, message: 'Thiếu câu hỏi' })
      }
    }
    setErrors(errs)
    setOk(errs.length ? '' : 'Validate OK — có thể publish.')
    return errs.length === 0
  }

  function save() {
    const idx = mockStore.drafts.findIndex((d) => d.id === draft.id)
    if (idx >= 0) {
      mockStore.drafts[idx] = {
        ...draft,
        updated_at: new Date().toISOString(),
      }
    }
    setOk('Đã lưu bản nháp (mock).')
  }

  function publish() {
    if (!validate()) return
    save()
    const version = '1.0.0'
    let proc = mockStore.procedures.find(
      (p) => p.procedure_code === def.procedure_code,
    )
    if (!proc) {
      proc = {
        id: `proc_${Date.now()}`,
        procedure_code: def.procedure_code,
        title: def.title,
        domain: draft.domain,
        active_version: version,
        status: 'ACTIVE',
        versions: [],
      }
      mockStore.procedures.unshift(proc)
    }
    proc.active_version = version
    proc.status = 'ACTIVE'
    proc.title = def.title
    proc.versions = [
      { version, status: 'ACTIVE', published_at: new Date().toISOString().slice(0, 10) },
      ...proc.versions.map((v) =>
        v.version === version ? v : { ...v, status: 'ARCHIVED' },
      ),
    ]
    draft.status = 'PUBLISHED'
    setOk(`Published ${def.procedure_code} @ ${version}`)
    setTimeout(() => navigate('/admin/procedures'), 600)
  }

  return (
    <div className="admin-page">
      <header className="admin-page-head">
        <p className="crumb">
          <Link to="/admin/drafts">Bản nháp</Link> / {draft.id}
        </p>
        <h1>Review: {def.title}</h1>
        <p>Sửa form theo definition — không chỉnh raw JSON.</p>
      </header>

      <div className="panel form-grid">
        <label>
          Mã thủ tục (procedure_code)
          <input
            value={def.procedure_code}
            onChange={(e) =>
              setDraft((p) => ({
                ...p,
                definition: { ...p.definition, procedure_code: e.target.value },
                procedure_code: e.target.value,
              }))
            }
          />
        </label>
        <label>
          Tiêu đề
          <input
            value={def.title}
            onChange={(e) =>
              setDraft((p) => ({
                ...p,
                title: e.target.value,
                definition: { ...p.definition, title: e.target.value },
              }))
            }
          />
        </label>
      </div>

      <div className="panel">
        <h2>Slots (câu hỏi)</h2>
        <div className="stack">
          {Object.entries(def.slots).map(([key, slot]) => (
            <div key={key} className="slot-card">
              <div className="slot-key">
                <code>{key}</code>
                <span className="pill">{slot.type}</span>
                {(def.required_slots || []).includes(key) ? (
                  <span className="pill warn">required</span>
                ) : null}
              </div>
              <label>
                Câu hỏi citizen
                <textarea
                  rows={2}
                  value={slot.question}
                  onChange={(e) => updateSlot(key, 'question', e.target.value)}
                />
              </label>
            </div>
          ))}
        </div>
      </div>

      <div className="panel form-grid">
        <h2>Guidance</h2>
        <label>
          Checklist (mỗi dòng 1 mục)
          <textarea
            rows={4}
            value={(def.guidance?.checklist || []).join('\n')}
            onChange={(e) => updateGuidance('checklist', e.target.value)}
          />
        </label>
        <label>
          Nơi nộp
          <input
            value={def.guidance?.where_to_submit || ''}
            onChange={(e) => updateGuidance('where_to_submit', e.target.value)}
          />
        </label>
      </div>

      {errors.length > 0 ? (
        <div className="panel danger">
          <h2>Lỗi validate</h2>
          <ul>
            {errors.map((err) => (
              <li key={err.path}>
                <code>{err.path}</code>: {err.message}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {ok ? <p className="form-ok">{ok}</p> : null}

      <div className="action-bar">
        <button type="button" onClick={save}>
          Lưu
        </button>
        <button type="button" onClick={validate}>
          Validate
        </button>
        <button type="button" className="btn-primary" onClick={publish}>
          Publish
        </button>
      </div>
    </div>
  )
}
