/**
 * After production build (VITE_ADMIN_INGESTION=false, no VITE_DEMO_LOGIN),
 * assert dist assets do not contain mockStore or demo credentials.
 */
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const dist = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist')

function walk(dir, acc = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) walk(p, acc)
    else if (/\.(js|css|html)$/.test(name)) acc.push(p)
  }
  return acc
}

if (!existsSync(dist)) {
  console.error('FAIL: dist/ missing — run production build before check-bundle')
  process.exit(1)
}

const files = walk(dist)
if (files.length === 0) {
  console.error('FAIL: dist/ has no assets')
  process.exit(1)
}

const joined = files.map((f) => readFileSync(f, 'utf8')).join('\n')
const indexingOn = process.argv.includes('--on')
const banned = indexingOn
  ? ['mockStore', 'admin@chuse.vn', 'admin123', 'citizen123', 'citizen@example.com', 'Activate']
  : [
      'mockStore',
      'admin@chuse.vn',
      'admin123',
      'citizen123',
      'citizen@example.com',
      'Activate',
      'Lập chỉ mục',
      'Thủ tục / phiên bản',
    ]
const required = indexingOn ? ['Lập chỉ mục', 'Thủ tục / phiên bản', 'Thử lại lập chỉ mục', 'Bỏ liên kết'] : []
let failed = false
for (const b of banned) {
  if (joined.includes(b)) {
    console.error('FAIL: production bundle contains', b)
    failed = true
  }
}
for (const needle of required) {
  if (!joined.includes(needle)) {
    console.error('FAIL: indexing bundle missing', needle)
    failed = true
  }
}
if (failed) process.exit(1)
console.log('production bundle hygiene: PASS (%d assets, indexing %s)', files.length, indexingOn ? 'on' : 'off')
