package session

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/chat"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/speech"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

type fakeTurns struct {
	err   error
	calls int
}

func (f *fakeTurns) Turn(context.Context, uuid.UUID, uuid.UUID, string) (*chat.TurnResult, error) {
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	return &chat.TurnResult{Action: "OUT_OF_SCOPE", ReplyText: "ok"}, nil
}

type linkCall struct {
	speechID uuid.UUID
	session  uuid.UUID
	turnID   uuid.UUID
}

type fakeLinker struct {
	mu      sync.Mutex
	calls   []linkCall
	err     error
	outcome speech.LinkOutcome
}

func (f *fakeLinker) LinkChatTurn(_ context.Context, speechRequestID, sessionID, chatTurnRequestID uuid.UUID) (speech.LinkResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, linkCall{speechID: speechRequestID, session: sessionID, turnID: chatTurnRequestID})
	out := f.outcome
	if out == "" {
		if f.err != nil {
			out = speech.LinkFailed
		} else {
			out = speech.LinkLinked
		}
	}
	return speech.LinkResult{Outcome: out}, f.err
}

func TestTurnLinksExplicitSpeechRequestID(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tok := "guest-link"
	sess := &repository.Session{ID: uuid.New(), GuestToken: &tok, XaID: "xa_a", Status: "OPEN"}
	turns := &fakeTurns{}
	linker := &fakeLinker{}
	h := &Handler{Sessions: sessionsOf(sess), Turns: turns, Speech: linker}
	speechID := uuid.New()
	turnID := uuid.New()
	res := chatPostTurn(t, h, sess.ID, turnID, `{"message":"tôi muốn khai sinh","speech_request_id":"`+speechID.String()+`"}`, map[string]string{HeaderGuestToken: tok})
	if res.Code != http.StatusOK {
		t.Fatalf("%d %s", res.Code, res.Body.String())
	}
	if turns.calls != 1 {
		t.Fatalf("turn calls %d", turns.calls)
	}
	if len(linker.calls) != 1 {
		t.Fatalf("link calls %+v", linker.calls)
	}
	got := linker.calls[0]
	if got.speechID != speechID || got.session != sess.ID || got.turnID != turnID {
		t.Fatalf("linked %+v", got)
	}
}

func TestTurnRetrySameSpeechAndTurnStillSucceeds(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tok := "guest-retry"
	sess := &repository.Session{ID: uuid.New(), GuestToken: &tok, XaID: "xa_a", Status: "OPEN"}
	turns := &fakeTurns{}
	linker := &fakeLinker{outcome: speech.LinkAlreadyLinked}
	h := &Handler{Sessions: sessionsOf(sess), Turns: turns, Speech: linker}
	speechID := uuid.New()
	turnID := uuid.New()
	body := `{"message":"tôi muốn khai sinh","speech_request_id":"` + speechID.String() + `"}`
	first := chatPostTurn(t, h, sess.ID, turnID, body, map[string]string{HeaderGuestToken: tok})
	retry := chatPostTurn(t, h, sess.ID, turnID, body, map[string]string{HeaderGuestToken: tok})
	if first.Code != http.StatusOK || retry.Code != http.StatusOK {
		t.Fatalf("retry changed turn HTTP: %d %d", first.Code, retry.Code)
	}
	if turns.calls != 2 {
		t.Fatalf("turn calls %d", turns.calls)
	}
	if len(linker.calls) != 2 {
		t.Fatalf("link calls %+v", linker.calls)
	}
	if linker.calls[0] != linker.calls[1] {
		t.Fatalf("retry swapped ids %+v", linker.calls)
	}
}

func TestTurnDoesNotInventSpeechLink(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tok := "guest-typed"
	sess := &repository.Session{ID: uuid.New(), GuestToken: &tok, XaID: "xa_a", Status: "OPEN"}
	linker := &fakeLinker{}
	h := &Handler{Sessions: sessionsOf(sess), Turns: &fakeTurns{}, Speech: linker}
	res := chatPostTurn(t, h, sess.ID, uuid.New(), `{"message":"xin chào cán bộ"}`, map[string]string{HeaderGuestToken: tok})
	if res.Code != http.StatusOK {
		t.Fatalf("%d %s", res.Code, res.Body.String())
	}
	if len(linker.calls) != 0 {
		t.Fatalf("typed chat linked %+v", linker.calls)
	}
}

func TestTurnIgnoresInvalidSpeechRequestID(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tok := "guest-bad"
	sess := &repository.Session{ID: uuid.New(), GuestToken: &tok, XaID: "xa_a", Status: "OPEN"}
	linker := &fakeLinker{}
	h := &Handler{Sessions: sessionsOf(sess), Turns: &fakeTurns{}, Speech: linker}
	res := chatPostTurn(t, h, sess.ID, uuid.New(), `{"message":"xin chào","speech_request_id":"not-a-uuid"}`, map[string]string{HeaderGuestToken: tok})
	if res.Code != http.StatusOK {
		t.Fatalf("%d %s", res.Code, res.Body.String())
	}
	if len(linker.calls) != 0 {
		t.Fatalf("invalid id linked %+v", linker.calls)
	}
}

func TestFailedTurnDoesNotLinkSpeech(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tok := "guest-fail"
	sess := &repository.Session{ID: uuid.New(), GuestToken: &tok, XaID: "xa_a", Status: "OPEN"}
	linker := &fakeLinker{}
	h := &Handler{
		Sessions: sessionsOf(sess),
		Turns:    &fakeTurns{err: repository.ErrSessionNotOpen},
		Speech:   linker,
	}
	res := chatPostTurn(t, h, sess.ID, uuid.New(), `{"message":"xin chào","speech_request_id":"`+uuid.NewString()+`"}`, map[string]string{HeaderGuestToken: tok})
	assertChatCode(t, res, http.StatusConflict, "CONFLICT")
	if len(linker.calls) != 0 {
		t.Fatalf("failed turn linked %+v", linker.calls)
	}
}

func TestSpeechLinkFailureDoesNotFailTurn(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tok := "guest-audit"
	sess := &repository.Session{ID: uuid.New(), GuestToken: &tok, XaID: "xa_a", Status: "OPEN"}
	logs := &bytes.Buffer{}
	h := &Handler{
		Sessions: sessionsOf(sess),
		Turns:    &fakeTurns{},
		Speech:   &fakeLinker{err: context.DeadlineExceeded, outcome: speech.LinkFailed},
		Logger:   slog.New(slog.NewTextHandler(logs, nil)),
	}
	res := chatPostTurn(t, h, sess.ID, uuid.New(), `{"message":"xin chào","speech_request_id":"`+uuid.NewString()+`"}`, map[string]string{HeaderGuestToken: tok})
	if res.Code != http.StatusOK {
		t.Fatalf("link error became citizen error: %d %s", res.Code, res.Body.String())
	}
	if !strings.Contains(logs.String(), "speech_chat_link") || !strings.Contains(logs.String(), "FAILED") {
		t.Fatalf("missing linkage log: %s", logs.String())
	}
	if strings.Contains(logs.String(), tok) || strings.Contains(logs.String(), "xin chào") {
		t.Fatalf("linkage log leaked token or message: %s", logs.String())
	}
}

func TestSpeechLinkNotFoundIsLoggedAndTurnSucceeds(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tok := "guest-miss"
	sess := &repository.Session{ID: uuid.New(), GuestToken: &tok, XaID: "xa_a", Status: "OPEN"}
	logs := &bytes.Buffer{}
	h := &Handler{
		Sessions: sessionsOf(sess),
		Turns:    &fakeTurns{},
		Speech:   &fakeLinker{outcome: speech.LinkNotFound},
		Logger:   slog.New(slog.NewTextHandler(logs, nil)),
	}
	res := chatPostTurn(t, h, sess.ID, uuid.New(), `{"message":"xin chào","speech_request_id":"`+uuid.NewString()+`"}`, map[string]string{HeaderGuestToken: tok})
	if res.Code != http.StatusOK {
		t.Fatalf("%d %s", res.Code, res.Body.String())
	}
	if !strings.Contains(logs.String(), "NOT_FOUND") {
		t.Fatalf("missing NOT_FOUND log: %s", logs.String())
	}
}

func chatPostTurn(t *testing.T, h *Handler, sessionID, turnID uuid.UUID, body string, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	engine := gin.New()
	engine.Use(middleware.RequestID())
	engine.POST("/api/v1/sessions/:sessionId/turns", h.Turn)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/sessions/"+sessionID.String()+"/turns", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(middleware.HeaderRequestID, turnID.String())
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	res := httptest.NewRecorder()
	engine.ServeHTTP(res, req)
	return res
}
