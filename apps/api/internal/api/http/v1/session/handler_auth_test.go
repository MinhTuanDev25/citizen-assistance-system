package session

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func TestChatGuestCannotUseOtherCommuneSession(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tokA, tokB := "guest-a", "guest-b"
	sessA := &repository.Session{ID: uuid.New(), GuestToken: &tokA, XaID: "xa_a", Status: "OPEN"}
	sessB := &repository.Session{ID: uuid.New(), GuestToken: &tokB, XaID: "xa_b", Status: "OPEN"}
	h := &Handler{Sessions: sessionsOf(sessA, sessB)}
	res := chatGetMessages(t, h, sessB.ID, map[string]string{HeaderGuestToken: tokA})
	assertChatCode(t, res, http.StatusForbidden, "FORBIDDEN")
}

func TestChatJWTCannotUseOtherUserSessionInOtherCommune(t *testing.T) {
	gin.SetMode(gin.TestMode)
	userA, userB := uuid.New(), uuid.New()
	sessB := &repository.Session{ID: uuid.New(), UserID: &userB, XaID: "xa_b", Status: "OPEN"}
	h := &Handler{Sessions: sessionsOf(sessB)}
	res := chatGetMessages(t, h, sessB.ID, map[string]string{middleware.ContextUserID: userA.String()})
	assertChatCode(t, res, http.StatusForbidden, "FORBIDDEN")
}

func TestChatSameUserMayAccessOwnedSessionInOtherCommune(t *testing.T) {
	gin.SetMode(gin.TestMode)
	uid := uuid.New()
	sessA := &repository.Session{ID: uuid.New(), UserID: &uid, XaID: "xa_a", Status: "OPEN"}
	sessB := &repository.Session{ID: uuid.New(), UserID: &uid, XaID: "xa_b", Status: "OPEN"}
	h := &Handler{Sessions: sessionsOf(sessA, sessB)}
	res := chatGetMessages(t, h, sessB.ID, map[string]string{middleware.ContextUserID: uid.String()})
	if res.Code != http.StatusOK {
		t.Fatalf("%d %s", res.Code, res.Body.String())
	}
	if !strings.Contains(res.Body.String(), `"xa_id":"xa_b"`) {
		t.Fatalf("expected owned session xa_b: %s", res.Body.String())
	}
}

func TestChatGuestCanAccessOwnSession(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tok := "guest-own"
	sess := &repository.Session{ID: uuid.New(), GuestToken: &tok, XaID: "xa_a", Status: "OPEN"}
	h := &Handler{Sessions: sessionsOf(sess)}
	res := chatGetMessages(t, h, sess.ID, map[string]string{HeaderGuestToken: tok})
	if res.Code != http.StatusOK {
		t.Fatalf("%d %s", res.Code, res.Body.String())
	}
}

func TestChatUnknownSessionIsNotCrossCommuneEvidence(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tok := "guest-own"
	sess := &repository.Session{ID: uuid.New(), GuestToken: &tok, XaID: "xa_a", Status: "OPEN"}
	h := &Handler{Sessions: sessionsOf(sess)}
	res := chatGetMessages(t, h, uuid.New(), map[string]string{HeaderGuestToken: tok})
	assertChatCode(t, res, http.StatusNotFound, "NOT_FOUND")
}

func chatGetMessages(t *testing.T, h *Handler, sessionID uuid.UUID, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	engine := gin.New()
	engine.Use(middleware.RequestID())
	if uid := headers[middleware.ContextUserID]; uid != "" {
		engine.Use(func(c *gin.Context) { c.Set(middleware.ContextUserID, uid); c.Next() })
	}
	engine.GET("/api/v1/sessions/:sessionId/messages", h.ListMessages)
	req := httptest.NewRequest(http.MethodGet, "/api/v1/sessions/"+sessionID.String()+"/messages", nil)
	for k, v := range headers {
		if k == middleware.ContextUserID {
			continue
		}
		req.Header.Set(k, v)
	}
	res := httptest.NewRecorder()
	engine.ServeHTTP(res, req)
	return res
}

func assertChatCode(t *testing.T, res *httptest.ResponseRecorder, status int, code string) {
	t.Helper()
	if res.Code != status {
		t.Fatalf("status %d want %d body %s", res.Code, status, res.Body.String())
	}
	var env map[string]any
	if err := json.Unmarshal(res.Body.Bytes(), &env); err != nil {
		t.Fatal(err)
	}
	errObj, _ := env["error"].(map[string]any)
	if fmt.Sprint(errObj["code"]) != code {
		t.Fatalf("code %v want %s", errObj["code"], code)
	}
}

type fakeChatSessions struct {
	byID map[uuid.UUID]*repository.Session
}

func sessionsOf(ss ...*repository.Session) *fakeChatSessions {
	m := make(map[uuid.UUID]*repository.Session, len(ss))
	for _, s := range ss {
		m[s.ID] = s
	}
	return &fakeChatSessions{byID: m}
}

func (f *fakeChatSessions) GetByID(_ context.Context, id uuid.UUID) (*repository.Session, error) {
	if f == nil || f.byID == nil {
		return nil, nil
	}
	return f.byID[id], nil
}

func (f *fakeChatSessions) GetByGuestToken(context.Context, string) (*repository.Session, error) {
	return nil, nil
}

func (f *fakeChatSessions) CreateForUser(context.Context, string, uuid.UUID) (*repository.Session, error) {
	return nil, nil
}

func (f *fakeChatSessions) CreateGuest(context.Context, string, string) (*repository.Session, error) {
	return nil, nil
}

func (f *fakeChatSessions) ListMessages(context.Context, uuid.UUID, int) ([]repository.Message, error) {
	return []repository.Message{}, nil
}
