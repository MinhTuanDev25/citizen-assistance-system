import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AuthProvider } from '../auth/AuthContext.jsx'
import RegisterPage, { validateRegistration } from './RegisterPage.jsx'

function setValue(input, value) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set
  setter.call(input, value)
  input.dispatchEvent(new Event('input', { bubbles: true }))
}

async function flush() {
  for (let i = 0; i < 4; i += 1) {
    await act(async () => { await Promise.resolve() })
  }
}

async function render() {
  const host = document.createElement('div')
  document.body.appendChild(host)
  await act(async () => {
    createRoot(host).render(createElement(AuthProvider, null,
      createElement(MemoryRouter, { initialEntries: ['/register'] },
        createElement(Routes, null,
          createElement(Route, { path: '/register', element: createElement(RegisterPage) }),
          createElement(Route, { path: '/', element: createElement('p', null, 'trang chủ') })))))
  })
  return host
}

async function fill(host, { fullName, email, password, confirm }) {
  const [name, mail, pw, pw2] = host.querySelectorAll('input')
  await act(async () => {
    setValue(name, fullName)
    setValue(mail, email)
    setValue(pw, password)
    setValue(pw2, confirm)
  })
}

async function submit(host) {
  await act(async () => { host.querySelector('button[type="submit"]').click() })
  await flush()
}

describe('RegisterPage', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
    localStorage.clear()
  })
  afterEach(() => vi.unstubAllGlobals())

  it('checks the fields like the server does', () => {
    const ok = { fullName: 'Nguyễn Văn A', email: 'a@b.vn', password: '123456', confirm: '123456' }
    expect(validateRegistration(ok)).toBe('')
    expect(validateRegistration({ ...ok, fullName: ' A ' })).toContain('họ tên')
    expect(validateRegistration({ ...ok, email: 'ab.vn' })).toBe('Email không hợp lệ.')
    expect(validateRegistration({ ...ok, password: '12345', confirm: '12345' })).toContain('6 ký tự')
    expect(validateRegistration({ ...ok, confirm: '1234567' })).toBe('Mật khẩu nhập lại không khớp.')
  })

  it('registers, signs in and opens the home page', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({
      data: { access_token: 'tok', user: { id: 'u1', email: 'a@b.vn', full_name: 'Nguyễn Văn A', role: 'CITIZEN' } },
    }), { status: 201 }))
    vi.stubGlobal('fetch', fetchMock)
    const host = await render()
    await fill(host, { fullName: 'Nguyễn Văn A', email: 'a@b.vn', password: '123456', confirm: '123456' })
    await submit(host)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/v1/auth/register')
    expect(JSON.parse(init.body)).toEqual({ full_name: 'Nguyễn Văn A', email: 'a@b.vn', password: '123456' })
    expect(JSON.parse(localStorage.getItem('cas_auth'))).toMatchObject({ id: 'u1', role: 'CITIZEN', accessToken: 'tok' })
    expect(host.textContent).toContain('trang chủ')
  })

  it('shows a taken email in Vietnamese', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ error: { code: 'CONFLICT', message: 'email already registered' } }), { status: 409 })))
    const host = await render()
    await fill(host, { fullName: 'Nguyễn Văn A', email: 'a@b.vn', password: '123456', confirm: '123456' })
    await submit(host)
    expect(host.querySelector('[role="alert"]').textContent).toBe('Email này đã được đăng ký. Hãy đăng nhập.')
    expect(localStorage.getItem('cas_auth')).toBeNull()
  })

  it('does not call the server when the passwords differ', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    const host = await render()
    await fill(host, { fullName: 'Nguyễn Văn A', email: 'a@b.vn', password: '123456', confirm: '654321' })
    await submit(host)
    expect(fetchMock).not.toHaveBeenCalled()
    expect(host.querySelector('[role="alert"]').textContent).toBe('Mật khẩu nhập lại không khớp.')
  })
})
