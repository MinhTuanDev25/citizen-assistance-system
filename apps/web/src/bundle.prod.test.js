import { describe, expect, it } from 'vitest'
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

describe('bundle contract pointer', () => {
  it('check-bundle script exists for CI', () => {
    const root = join(dirname(fileURLToPath(import.meta.url)), '..')
    const script = join(root, 'scripts', 'check-bundle.mjs')
    expect(existsSync(script)).toBe(true)
  })
})
