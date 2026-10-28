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

  it('rejects open source conflicts and unacknowledged coverage gaps', () => {
    const ok = {
      status: 'DRAFT',
      validation: { schema_valid: true, source_coverage_complete: true, source_quotes_verified: true, grounding_verified: true },
    }
    expect(canApproveDraft({ ...ok, conflicts: [{ status: 'OPEN' }] })).toBe(false)
    expect(canApproveDraft({ ...ok, conflicts: [{ status: 'RESOLVED' }] })).toBe(true)
    // A decision reopened on read (edited value, or stored before pinning) blocks again.
    expect(canApproveDraft({ ...ok, conflicts: [{ status: 'OPEN', reopened: true }] })).toBe(false)
    // A conflict of a document the Definition does not cite blocks nothing.
    expect(canApproveDraft({ ...ok, conflicts: [{ status: 'OPEN', field: 'citations', inactive: true }] })).toBe(true)
    expect(canApproveDraft({ ...ok, coverage: { gaps: [{ field: 'guidance.fee_hint' }] } })).toBe(false)
    expect(canApproveDraft({ ...ok, coverage: { gaps: [{ field: 'guidance.fee_hint', acknowledged: true }] } })).toBe(true)
  })

  it('rejects unreviewed truncation and unverified legal sources', () => {
    const ok = {
      status: 'DRAFT',
      validation: { schema_valid: true, source_coverage_complete: true, source_quotes_verified: true, grounding_verified: true },
    }
    const trunc = { field: 'guidance.checklist', blocking: true, dropped: ['x'] }
    expect(canApproveDraft({ ...ok, coverage: { truncated: [trunc] } })).toBe(false)
    expect(canApproveDraft({ ...ok, coverage: { truncated: [trunc], acknowledged: [{ field: 'truncated:guidance.checklist' }] } })).toBe(true)
    expect(canApproveDraft({ ...ok, coverage: { truncated: [{ ...trunc, blocking: false }] } })).toBe(true)
    expect(canApproveDraft({ ...ok, coverage: { outside_mentions_reviewed: false } })).toBe(false)
    expect(canApproveDraft({ ...ok, coverage: { outside_mentions_reviewed: true } })).toBe(true)
    expect(canApproveDraft({ ...ok, coverage: { legal_sources_verified: false } })).toBe(false)
    expect(canApproveDraft({ ...ok, coverage: { boundary_uncertain: true } })).toBe(false)
    expect(canApproveDraft({ ...ok, coverage: { boundary_uncertain: true, boundary_reviewed: true } })).toBe(true)
  })
})
