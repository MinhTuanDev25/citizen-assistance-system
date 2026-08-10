import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider } from './auth/AuthContext.jsx'
import { CommuneProvider } from './commune/CommuneContext.jsx'
import { RequireAdmin } from './components/RequireAdmin.jsx'
import { RequireCommune } from './components/RequireCommune.jsx'
import { AdminShell } from './components/AdminShell.jsx'
import CitizenChatPage from './pages/CitizenChatPage.jsx'
import LoginPage from './pages/LoginPage.jsx'
import SelectCommunePage from './pages/SelectCommunePage.jsx'
import AdminDashboardPage from './pages/admin/AdminDashboardPage.jsx'
import DocumentsPage from './pages/admin/DocumentsPage.jsx'
import DraftsPage from './pages/admin/DraftsPage.jsx'
import DraftReviewPage from './pages/admin/DraftReviewPage.jsx'
import ProceduresPage from './pages/admin/ProceduresPage.jsx'

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
              <Route index element={<AdminDashboardPage />} />
              <Route path="documents" element={<DocumentsPage />} />
              <Route path="drafts" element={<DraftsPage />} />
              <Route path="drafts/:draftId" element={<DraftReviewPage />} />
              <Route path="procedures" element={<ProceduresPage />} />
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </CommuneProvider>
    </AuthProvider>
  )
}
