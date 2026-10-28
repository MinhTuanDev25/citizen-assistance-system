/** Shared labels for places outside a draft's scope (plan preview and review). */

export const OUTSIDE_TITLE = 'Các nguồn có thể liên quan ngoài phạm vi đã soạn'

export function mentionLabel(m) {
  if (m?.kind === 'general') return 'Quy định chung (từ khóa lệ phí/thời hạn/biểu mẫu)'
  return m?.section || 'Ngoài các mục thủ tục — nhắc tên thủ tục'
}

/** Exact pages ("2, 100, 450" or "7–9"); falls back to bounds for older drafts. */
export function mentionPages(m) {
  const spans = m?.pages || []
  if (!spans.length) return `${m?.page_start}–${m?.page_end}`
  return spans.map((s) => (s.start === s.end ? `${s.start}` : `${s.start}–${s.end}`)).join(', ')
}

/**
 * Short label that tells two citations with the same title apart: the title,
 * then the document number, else a short source code (never a full UUID).
 */
export function citationLabel(c) {
  const title = String(c?.title || '').trim() || 'Văn bản không tên'
  const number = String(c?.document_number || '').trim()
  if (number) return `${title} (số ${number})`
  const id = String(c?.doc_id || c?.document_id || '').replace(/-/g, '').trim()
  return id ? `${title} · mã ${id.slice(0, 8)}` : title
}
