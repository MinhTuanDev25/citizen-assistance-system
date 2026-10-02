import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { lazy, Suspense } from 'react'
import { AuthProvider } from './auth/AuthContext.jsx'
import { CommuneProvider } from './commune/CommuneContext.jsx'
import { RequireAdmin } from './components/RequireAdmin.jsx'
import { RequireCommune } from './components/RequireCommune.jsx'
import { AdminShell } from './components/AdminShell.jsx'
import { adminIngestionEnabled } from './config/features.js'
import CitizenChatPage from './pages/CitizenChatPage.jsx'
import LoginPage from './pages/LoginPage.jsx'
import SelectCommunePage from './pages/SelectCommunePage.jsx'
import DeferredIngestionPage from './pages/admin/DeferredIngestionPage.jsx'
import ProceduresPage from './pages/admin/ProceduresPage.jsx'

// Build-time constant so Vite drops admin mock chunks when flag is false.
const ingestionOn = adminIngestionEnabled()

const AdminDashboardPage = ingestionOn
  ? lazy(() => import('./pages/admin/AdminDashboardPage.jsx'))
  : null
const DocumentsPage = ingestionOn
  ? lazy(() => import('./pages/admin/DocumentsPage.jsx'))
  : null

function SuspenseAdmin({ children, title }) {
  return (
    <Suspense fallback={<DeferredIngestionPage title={title} />}>{children}</Suspense>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <CommuneProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/chon-xa" element={<SelectCommunePage />} />
            <Route
              path="/"
              element={
                <RequireCommune>
                  <CitizenChatPage />
                </RequireCommune>
              }
            />
            <Route
              path="/login"
              element={
                <RequireCommune>
                  <LoginPage />
                </RequireCommune>
              }
            />
            <Route
              path="/admin"
              element={
                <RequireCommune>
                  <RequireAdmin>
                    <AdminShell />
                  </RequireAdmin>
                </RequireCommune>
              }
            >
              <Route
                index
                element={
                  ingestionOn && AdminDashboardPage ? (
                    <SuspenseAdmin title="Tổng quan admin">
                      <AdminDashboardPage />
                    </SuspenseAdmin>
                  ) : (
                    <DeferredIngestionPage title="Tổng quan admin" />
                  )
                }
              />
              <Route
                path="documents"
                element={
                  ingestionOn && DocumentsPage ? (
                    <SuspenseAdmin title="Tài liệu">
                      <DocumentsPage />
                    </SuspenseAdmin>
                  ) : (
                    <DeferredIngestionPage title="Tài liệu" />
                  )
                }
              />
              <Route
                path="drafts"
                element={<DeferredIngestionPage title="Bản nháp thủ tục" />}
              />
              <Route
                path="drafts/:draftId"
                element={<DeferredIngestionPage title="Review bản nháp" />}
              />
              <Route path="procedures" element={<ProceduresPage />} />
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </CommuneProvider>
    </AuthProvider>
  )
}
