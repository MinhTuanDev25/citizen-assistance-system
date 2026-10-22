import { describe, expect, it } from 'vitest'
import { canApproveDraft } from './drafts.js'

describe('canApproveDraft', () => {
  it('rejects incomplete coverage', () => {
    expect(canApproveDraft({
      status: 'DRAFT',
      validation: { schema_valid: true, source_coverage_complete: false, source_quotes_verified: true, grounding_verified: false },
    })).toBe(false)
  })

  it('rejects identity mismatch', () => {
    expect(canApproveDraft({
      status: 'DRAFT',
      identity_mismatch: true,
      validation: { schema_valid: false, source_coverage_complete: true, source_quotes_verified: true, grounding_verified: false },
    })).toBe(false)
  })

  it('allows a fully validated draft to request approval', () => {
    expect(canApproveDraft({
      status: 'DRAFT',
      validation: {
        schema_valid: true,
        source_coverage_complete: true,
        source_quotes_verified: true,
        grounding_verified: true,
      },
    })).toBe(true)
  })
})
