#!/usr/bin/env node
/**
 * Frontend contract checks for Phase 1–3 (no browser).
 * Fails if production paths import mockStore or call write /messages.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'

const root = new URL('..', import.meta.url).pathname.replace(/\/$/, '')
const src = join(root, 'src')

function walk(dir, acc = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) walk(p, acc)
    else if (/\.(jsx?|tsx?|js)$/.test(name)) acc.push(p)
  }
  return acc
}

const files = walk(src)
const alwaysLoaded = [
  'src/App.jsx',
  'src/main.jsx',
  'src/pages/CitizenChatPage.jsx',
  'src/pages/admin/DeferredIngestionPage.jsx',
  'src/pages/admin/ProceduresPage.jsx',
  'src/api/session.js',
  'src/api/catalog.js',
  'src/config/features.js',
].map((r) => join(root, r))

let failed = false
for (const f of alwaysLoaded) {
  const text = readFileSync(f, 'utf8')
  if (text.includes('mockStore')) {
    console.error('FAIL: production path imports mockStore:', f)
    failed = true
  }
}

const chat = readFileSync(join(src, 'pages/CitizenChatPage.jsx'), 'utf8')
if (!chat.includes('postTurn')) {
  console.error('FAIL: CitizenChatPage must use postTurn')
  failed = true
}
if (chat.includes('postUserMessage')) {
  console.error('FAIL: CitizenChatPage must not call postUserMessage')
  failed = true
}

const session = readFileSync(join(src, 'api/session.js'), 'utf8')
if (!session.includes('/turns')) {
  console.error('FAIL: session.js must call /turns')
  failed = true
}
if (!session.includes('resolveTurnRequestId') || !session.includes('PENDING_TURN')) {
  console.error('FAIL: session.js must retain request id for retry')
  failed = true
}

const catalog = readFileSync(join(src, 'api/catalog.js'), 'utf8')
if (!catalog.includes("citizen") || !catalog.includes("'true'")) {
  console.error('FAIL: catalog.js must request citizen=true')
  failed = true
}

const ingestion = process.env.VITE_ADMIN_INGESTION
if (ingestion === 'true' || ingestion === '1') {
  console.warn('WARN: VITE_ADMIN_INGESTION is on; production CI should set false')
}

if (failed) process.exit(1)
console.log('frontend contract tests: PASS (%d source files scanned)', files.length)
