import { describe, expect, it } from 'vitest'
import {
  citationExcerpt,
  citationPage,
  citationPdfHref,
  citationTitle,
  indexedPdfCitations,
  retrievalHint,
  splitEvidence,
} from './citations.js'

describe('citation helpers', () => {
  it('keeps only indexed PDF rows', () => {
    const rows = indexedPdfCitations([
      { source_type: 'manual_seed', title: 'seed' },
      { source_type: 'indexed_pdf', title: 'pdf', excerpt: '<b>raw</b>' },
    ])
    expect(rows).toHaveLength(1)
    expect(citationTitle(rows[0])).toBe('pdf')
    expect(citationExcerpt(rows[0])).toBe('<b>raw</b>')
  })

  it('fails closed: missing status is related, not supporting', () => {
    const { supporting, related } = splitEvidence([
      { source_type: 'indexed_pdf', title: 'A', evidence_status: 'supporting', page_start: 3 },
      { source_type: 'indexed_pdf', title: 'B' },
    ])
    expect(supporting).toHaveLength(1)
    expect(related).toHaveLength(1)
    expect(citationPage(supporting[0])).toBe('3')
  })

  it('builds a same-origin PDF href with page fragment', () => {
    expect(
      citationPdfHref({
        source_type: 'indexed_pdf',
        document_id: '60b677df-acc0-4198-a3b6-a919c01482a1',
        page_start: 11,
        availability: 'available',
      }),
    ).toBe('/api/v1/documents/60b677df-acc0-4198-a3b6-a919c01482a1/content#page=11')
  })

  it('does not invent PDF links for seed or unavailable citations', () => {
    expect(citationPdfHref({ source_type: 'manual_seed', doc_id: 'seed_dk_khai_sinh_v1' })).toBe('')
    expect(
      citationPdfHref({
        document_id: '60b677df-acc0-4198-a3b6-a919c01482a1',
        availability: 'unavailable',
        page_start: 11,
      }),
    ).toBe('')
  })

  it('does not invent a verified empty state', () => {
    expect(indexedPdfCitations(null)).toEqual([])
    expect(retrievalHint('')).toBe('')
    expect(retrievalHint('ok')).toBe('')
    expect(retrievalHint('no_relevant_evidence')).toMatch(/Chưa tìm được/)
    expect(retrievalHint('all_candidates_rejected')).toMatch(/không khớp/)
    expect(retrievalHint('retrieval_unavailable')).toMatch(/Không kiểm tra/)
    expect(retrievalHint('retrieval_incompatible')).toMatch(/Không kiểm tra/)
    expect(retrievalHint('ok', 'allowed_sources_truncated')).toMatch(/một phần nguồn/)
  })
})
