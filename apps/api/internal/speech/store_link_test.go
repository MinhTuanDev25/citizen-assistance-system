package speech

import (
	"context"
	"errors"
	"testing"

	"github.com/google/uuid"
)

func TestLinkChatTurnNilStoreIsFailedNotSilent(t *testing.T) {
	t.Parallel()
	id := uuid.New()
	out, err := (*Store)(nil).LinkChatTurn(context.Background(), id, id, id)
	if out.Outcome != LinkFailed {
		t.Fatalf("outcome %s", out.Outcome)
	}
	if !errors.Is(err, ErrAuditUnavailable) {
		t.Fatalf("err %v", err)
	}
	out, err = (&Store{}).LinkChatTurn(context.Background(), id, id, id)
	if out.Outcome != LinkFailed || !errors.Is(err, ErrAuditUnavailable) {
		t.Fatalf("empty pool %s %v", out.Outcome, err)
	}
}
