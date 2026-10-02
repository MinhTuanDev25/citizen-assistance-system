package index

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/google/uuid"
)

type fakeStore struct {
	claim  ClaimResult
	finish FinishInput
}

func (f *fakeStore) ListTargets(context.Context, string) ([]Target, error) { return nil, nil }
func (f *fakeStore) ListLinks(context.Context, string, uuid.UUID) ([]Link, error) {
	return nil, nil
}
func (f *fakeStore) Link(context.Context, string, LinkInput, string) (Link, bool, error) {
	return Link{}, false, nil
}
func (f *fakeStore) Unlink(context.Context, string, uuid.UUID, uuid.UUID, uuid.UUID, uuid.UUID, string) (bool, error) {
	return false, nil
}
func (f *fakeStore) Claim(context.Context, ClaimInput) (ClaimResult, error) { return f.claim, nil }
func (f *fakeStore) StageGeneration(context.Context, StageInput) (StagedObject, error) {
	return StagedObject{}, nil
}
func (f *fakeStore) Finish(_ context.Context, in FinishInput) (FinishResult, error) {
	f.finish = in
	return FinishResult{Applied: true, LinkStatus: in.Outcome, DocumentStatus: in.Outcome}, nil
}

func TestRequestSuccessDoesNotInventChunks(t *testing.T) {
	docID := uuid.New()
	ver := uuid.New()
	store := &fakeStore{claim: ClaimResult{
		JobID: uuid.New(), ClaimToken: uuid.New(), RunWorker: true,
		WorkerRequest: Request{SchemaVersion: SchemaVersion, DocumentID: docID, ProcedureVersionID: ver},
	}}
	svc := &Service{
		XAID: "xa_chu_se", Store: store, Timeout: time.Second, ClaimTTL: 5 * time.Second,
		Worker: WorkerFunc(func(context.Context, Request) (Response, error) {
			return Response{SchemaVersion: SchemaVersion, DocumentID: docID, ProcedureVersionID: ver, Outcome: OutcomeReady}, nil
		}),
	}
	got, err := svc.Request(context.Background(), docID, ver, uuid.New(), uuid.New(), false)
	if err != nil {
		t.Fatal(err)
	}
	if got.LinkStatus != OutcomeReady || store.finish.Outcome != OutcomeReady {
		t.Fatalf("status %s finish %s", got.LinkStatus, store.finish.Outcome)
	}
}

func TestRequestTimeoutMarksFailed(t *testing.T) {
	docID := uuid.New()
	ver := uuid.New()
	store := &fakeStore{claim: ClaimResult{JobID: uuid.New(), ClaimToken: uuid.New(), RunWorker: true}}
	svc := &Service{
		XAID: "xa_chu_se", Store: store, Timeout: 20 * time.Millisecond, ClaimTTL: 5 * time.Second,
		Worker: WorkerFunc(func(ctx context.Context, _ Request) (Response, error) {
			<-ctx.Done()
			return Response{}, ctx.Err()
		}),
	}
	got, err := svc.Request(context.Background(), docID, ver, uuid.New(), uuid.New(), true)
	if err != nil {
		t.Fatal(err)
	}
	if got.LinkStatus != OutcomeFailed || got.ErrorCode != "timeout" || store.finish.Outcome != OutcomeFailed {
		t.Fatalf("%+v finish %+v", got, store.finish)
	}
}

func TestWorkerVersionMismatchIsFailed(t *testing.T) {
	docID := uuid.New()
	ver := uuid.New()
	store := &fakeStore{claim: ClaimResult{JobID: uuid.New(), ClaimToken: uuid.New(), RunWorker: true}}
	svc := &Service{
		XAID: "xa_chu_se", Store: store, Timeout: time.Second, ClaimTTL: 5 * time.Second,
		Worker: WorkerFunc(func(context.Context, Request) (Response, error) {
			return Response{SchemaVersion: SchemaVersion, DocumentID: docID, ProcedureVersionID: uuid.New(), Outcome: OutcomeReady}, nil
		}),
	}
	got, err := svc.Request(context.Background(), docID, ver, uuid.New(), uuid.New(), false)
	if err != nil {
		t.Fatal(err)
	}
	if got.ErrorCode != "worker_mismatch" || store.finish.Outcome != OutcomeFailed {
		t.Fatalf("%+v %+v", got, store.finish)
	}
}

func TestRetryFromReadyIsRejectedByStore(t *testing.T) {
	store := &rejectStore{err: ErrInvalidState}
	svc := &Service{XAID: "xa_chu_se", Store: store, Worker: WorkerFunc(func(context.Context, Request) (Response, error) {
		t.Fatal("worker ran")
		return Response{}, errors.New("nope")
	})}
	_, err := svc.Request(context.Background(), uuid.New(), uuid.New(), uuid.New(), uuid.New(), true)
	if !errors.Is(err, ErrInvalidState) {
		t.Fatal(err)
	}
}

func TestPageRangeRejectsReversedBounds(t *testing.T) {
	if _, err := cleanPageRange("2-1"); err == nil {
		t.Fatal("reversed range accepted")
	}
	if _, err := cleanPageRange("1-2"); err != nil {
		t.Fatal(err)
	}
}

type rejectStore struct{ err error }

func (r rejectStore) ListTargets(context.Context, string) ([]Target, error) { return nil, nil }
func (r rejectStore) ListLinks(context.Context, string, uuid.UUID) ([]Link, error) {
	return nil, nil
}
func (r rejectStore) Link(context.Context, string, LinkInput, string) (Link, bool, error) {
	return Link{}, false, nil
}
func (r rejectStore) Unlink(context.Context, string, uuid.UUID, uuid.UUID, uuid.UUID, uuid.UUID, string) (bool, error) {
	return false, nil
}
func (r rejectStore) Claim(context.Context, ClaimInput) (ClaimResult, error) {
	return ClaimResult{}, r.err
}
func (r rejectStore) StageGeneration(context.Context, StageInput) (StagedObject, error) {
	return StagedObject{}, nil
}
func (r rejectStore) Finish(context.Context, FinishInput) (FinishResult, error) {
	return FinishResult{}, nil
}

type WorkerFunc func(context.Context, Request) (Response, error)

func (f WorkerFunc) Index(ctx context.Context, req Request) (Response, error) { return f(ctx, req) }
