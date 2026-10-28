package drafting

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/uuid"
)

func TestDraftTimeoutFromAIServiceOrDeadline(t *testing.T) {
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusGatewayTimeout)
		_, _ = w.Write([]byte(`{"detail":{"reason":"timeout"}}`))
	}))
	defer gateway.Close()
	if _, err := NewClient(gateway.URL, time.Second).Draft(context.Background(), uuid.New(), SourceSnapshot{}); !errors.Is(err, ErrDraftTimeout) {
		t.Fatalf("AI 504: want ErrDraftTimeout, got %v", err)
	}

	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Outlive the client deadline, but end on its own so Close does not wait on a kept-alive conn.
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
	}))
	defer slow.Close()
	if _, err := NewClient(slow.URL, 50*time.Millisecond).Draft(context.Background(), uuid.New(), SourceSnapshot{}); !errors.Is(err, ErrDraftTimeout) {
		t.Fatalf("client deadline: want ErrDraftTimeout, got %v", err)
	}
}
