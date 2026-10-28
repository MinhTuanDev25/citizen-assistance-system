import { describe, expect, it } from 'vitest'
import { citationLabel, mentionPages } from './draftSources.js'

describe('draftSources', () => {
  it('labels citations so two with the same title differ', () => {
    expect(citationLabel({ title: 'Quyết định 163', document_number: '163/QĐ-BTP', doc_id: 'x' })).toBe('Quyết định 163 (số 163/QĐ-BTP)')
    expect(citationLabel({ title: 'Quyết định 163', doc_id: '8939d98f-ce2c-487f-9f28-a53510c29271' })).toBe('Quyết định 163 · mã 8939d98f')
    expect(citationLabel({ title: 'Quyết định 163', document_id: 'abcdef1234' })).toBe('Quyết định 163 · mã abcdef12')
    expect(citationLabel({ title: '  ' })).toBe('Văn bản không tên')
  })

  it('formats exact page spans', () => {
    expect(mentionPages({ pages: [{ start: 2, end: 2 }, { start: 100, end: 101 }] })).toBe('2, 100–101')
    expect(mentionPages({ page_start: 1, page_end: 8 })).toBe('1–8')
  })
})
