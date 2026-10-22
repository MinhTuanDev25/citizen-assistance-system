function field(cite, ...keys) {
  for (const key of keys) {
    const value = cite?.[key]
    if (value !== undefined && value !== null && String(value).trim() !== '') {
      return value
    }
  }
  return ''
}

export function indexedPdfCitations(citations) {
  return (Array.isArray(citations) ? citations : []).filter((cite) => {
    const type = field(cite, 'source_type', 'SourceType')
    return type === 'indexed_pdf'
  })
}

export function splitEvidence(citations) {
  const pdf = indexedPdfCitations(citations)
  const supporting = pdf.filter((cite) => field(cite, 'evidence_status', 'EvidenceStatus') === 'supporting')
  const related = pdf.filter((cite) => field(cite, 'evidence_status', 'EvidenceStatus') !== 'supporting')
  return { supporting, related }
}

export function citationTitle(cite) {
  return String(field(cite, 'title', 'Title') || 'Tài liệu PDF')
}

export function citationPage(cite) {
  const hint = field(cite, 'page_hint', 'PageHint', 'page_range')
  if (hint) return String(hint)
  const start = Number(cite?.page_start || cite?.PageStart || 0)
  const end = Number(cite?.page_end || cite?.PageEnd || 0)
  if (!start) return ''
  if (end > start) return `${start}–${end}`
  return String(start)
}

export function citationExcerpt(cite) {
  return String(field(cite, 'excerpt', 'Excerpt'))
}

export function citationAvailability(cite) {
  const value = field(cite, 'availability', 'Availability') || 'available'
  if (value === 'archived') return 'Nguồn đã lưu trữ'
  if (value === 'unavailable') return 'Tệp gốc không còn'
  return ''
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

export function citationPdfHref(cite) {
  const id = String(field(cite, 'document_id', 'DocumentID'))
  if (!UUID_RE.test(id)) return ''
  const availability = field(cite, 'availability', 'Availability') || 'available'
  if (availability === 'unavailable') return ''
  const start = Number(cite?.page_start || cite?.PageStart || 0)
  const page = Number.isFinite(start) && start > 0 ? `#page=${start}` : ''
  return `/api/v1/documents/${id}/content${page}`
}

export function retrievalHint(status, reason) {
  if (reason === 'allowed_sources_truncated') {
    return 'Chỉ đối chiếu một phần nguồn đã ghim. Checklist vẫn theo định nghĩa thủ tục.'
  }
  if (status === 'retrieval_unavailable' || status === 'retrieval_incompatible') {
    return 'Không kiểm tra được nguồn PDF lúc này. Hướng dẫn thủ tục vẫn theo định nghĩa đã kích hoạt.'
  }
  if (status === 'all_candidates_rejected') {
    return 'Đã tìm thấy đoạn PDF nhưng không dùng được vì dữ liệu nguồn không khớp. Checklist vẫn theo định nghĩa thủ tục.'
  }
  if (status === 'no_relevant_evidence') {
    return 'Chưa tìm được đoạn PDF đối chiếu cho nội dung này. Checklist vẫn theo định nghĩa thủ tục.'
  }
  return ''
}
