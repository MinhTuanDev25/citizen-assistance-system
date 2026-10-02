package document

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log/slog"
	"testing"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/storage"
	"github.com/google/uuid"
)

type cancelOnInsert struct {
	cancel context.CancelFunc
}

func (c cancelOnInsert) Insert(context.Context, Document, uuid.UUID) error {
	c.cancel()
	return errors.New("insert failed")
}

func (cancelOnInsert) Get(context.Context, string, uuid.UUID) (Document, error) {
	return Document{}, ErrNotFound
}

func (cancelOnInsert) List(context.Context, ListFilter) (ListResult, error) {
	return ListResult{}, nil
}

func (cancelOnInsert) DomainActive(context.Context, string) error { return nil }

type ctxStore struct {
	*storage.Memory
	deleteErr error
	deleted   bool
}

func (s *ctxStore) Delete(ctx context.Context, key string) error {
	s.deleteErr = ctx.Err()
	if err := ctx.Err(); err != nil {
		return err
	}
	s.deleted = true
	return s.Memory.Delete(ctx, key)
}

func TestInsertFailureCleansObjectAfterCanceledRequest(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	objects := &ctxStore{Memory: storage.NewMemory()}
	svc := &Service{
		Repo:     cancelOnInsert{cancel: cancel},
		Objects:  objects,
		Bucket:   "cas-documents",
		XAID:     "xa_chu_se",
		MaxBytes: 1024,
		TempDir:  t.TempDir(),
		Logger:   slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	raw := append([]byte("%PDF-1.4\n"), []byte("orphan")...)
	prep, err := svc.Prepare(bytes.NewReader(raw), "a.pdf", MimePDF, -1)
	if err != nil {
		t.Fatal(err)
	}
	_, err = svc.Save(ctx, prep, UploadMeta{
		Title:       "Giấy",
		DomainID:    "ho_tich_chung_thuc",
		ActorUserID: uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
		RequestID:   uuid.New(),
	})
	if err == nil {
		t.Fatal("expected insert failure")
	}
	if objects.Len() != 0 || !objects.deleted {
		t.Fatal("canceled request left an object behind")
	}
	if objects.deleteErr != nil {
		t.Fatal("cleanup used a canceled context")
	}
}
