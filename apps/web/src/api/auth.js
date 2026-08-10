import { apiRequest } from './client.js'

function mapUser(apiUser, accessToken) {
  if (!apiUser) return null
  return {
    id: apiUser.id,
    email: apiUser.email || '',
    name: apiUser.full_name || apiUser.name || '',
    role: apiUser.role,
    accessToken: accessToken || null,
  }
}

export async function loginRequest(email, password) {
  const data = await apiRequest('/api/v1/auth/login', {
    method: 'POST',
    auth: false,
    body: JSON.stringify({ email, password }),
  })
  return mapUser(data.user, data.access_token)
}

export async function registerRequest({ fullName, email, password }) {
  const data = await apiRequest('/api/v1/auth/register', {
    method: 'POST',
    auth: false,
    body: JSON.stringify({
      full_name: fullName,
      email,
      password,
    }),
  })
  return mapUser(data.user, data.access_token)
}

export async function meRequest() {
  const data = await apiRequest('/api/v1/auth/me')
  const auth = JSON.parse(localStorage.getItem('cas_auth') || 'null')
  return mapUser(data.user, auth?.accessToken)
}

export async function logoutRequest() {
  try {
    await apiRequest('/api/v1/auth/logout', { method: 'POST' })
  } catch {
    // Client still clears token even if request fails
  }
}
