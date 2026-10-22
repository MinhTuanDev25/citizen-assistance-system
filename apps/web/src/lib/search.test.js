import { describe, expect, it } from 'vitest'
import { foldQuery, matchesQuery } from './search.js'

describe('search matching', () => {
  it('ignores Vietnamese diacritics and extra case', () => {
    expect(foldQuery('Chứng thực')).toBe('chung thuc')
    expect(matchesQuery('chung', 'Chứng thực bản sao')).toBe(true)
    expect(matchesQuery('HI', 'hi', 'a.pdf')).toBe(true)
    expect(matchesQuery('xyz', 'Chứng thực')).toBe(false)
    expect(matchesQuery('  ', 'hi')).toBe(false)
  })
})
