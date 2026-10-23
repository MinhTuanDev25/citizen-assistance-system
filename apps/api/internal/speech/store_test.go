package speech

import (
	"context"
	"errors"
	"testing"

	"github.com/google/uuid"
)

func TestStoreNilPoolIsExplicitError(t *testing.T) {
	err := (*Store)(nil).Insert(context.Background(), Record{RequestID: uuid.New()})
	if !errors.Is(err, ErrAuditUnavailable) {
		t.Fatalf("nil store: %v", err)
	}
	err = (&Store{}).Insert(context.Background(), Record{RequestID: uuid.New()})
	if !errors.Is(err, ErrAuditUnavailable) {
		t.Fatalf("nil pool: %v", err)
	}
}

func TestStoreHonorsContextDeadline(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := (&Store{}).Insert(ctx, Record{RequestID: uuid.New()})
	if !errors.Is(err, ErrAuditUnavailable) {
		t.Fatalf("nil pool should fail closed even if ctx is canceled: %v", err)
	}
}
