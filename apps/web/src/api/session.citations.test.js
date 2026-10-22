import { describe, expect, it } from 'vitest'
import { mapApiMessage } from './session.js'

describe('mapApiMessage citations', () => {
  it('keeps historical excerpt and retrieval status from metadata', () => {
    const mapped = mapApiMessage({
      id: 'm1',
      role: 'ASSISTANT',
      content: 'Hướng dẫn chứng thực',
      action: 'DIRECT_ANSWER',
      created_at: '2026-10-15T00:00:00Z',
      message_metadata: {
        retrieval_status: 'ok',
        retrieval_reason: 'allowed_sources_truncated',
        citations: [
          {
            source_type: 'indexed_pdf',
            title: 'HD chứng thực PDF',
            page_hint: '3',
            excerpt: '<script>x</script> Bản chính giấy tờ cần chứng thực',
            evidence_status: 'supporting',
            generation_id: 'cccccccc-cccc-cccc-cccc-cccccccccccc',
          },
        ],
      },
    })
    expect(mapped.retrievalStatus).toBe('ok')
    expect(mapped.retrievalReason).toBe('allowed_sources_truncated')
    expect(mapped.citations[0].excerpt).toContain('<script>')
    expect(mapped.citations[0].generation_id).toBe('cccccccc-cccc-cccc-cccc-cccccccccccc')
  })

  it('does not invent citations', () => {
    const mapped = mapApiMessage({
      id: 'm2',
      role: 'ASSISTANT',
      content: 'xin chào',
      message_metadata: {},
    })
    expect(mapped.citations).toBeNull()
    expect(mapped.retrievalStatus).toBeNull()
  })
})
