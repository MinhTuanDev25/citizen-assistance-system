package index

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/google/uuid"
)

type fakeStore struct {
	claim     ClaimResult
	finish    FinishInput
	lastClaim ClaimInput
	claims    []ClaimInput
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
func (f *fakeStore) Claim(_ context.Context, in ClaimInput) (ClaimResult, error) {
	f.lastClaim = in
	f.claims = append(f.claims, in)
	return f.claim, nil
}
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

func TestRequestKeepsWorkerAfterCallerCancel(t *testing.T) {
	docID := uuid.New()
	ver := uuid.New()
	store := &fakeStore{claim: ClaimResult{
		JobID: uuid.New(), ClaimToken: uuid.New(), RunWorker: true,
		WorkerRequest: Request{SchemaVersion: SchemaVersion, DocumentID: docID, ProcedureVersionID: ver},
	}}
	started := make(chan struct{})
	svc := &Service{
		XAID: "xa_chu_se", Store: store, Timeout: time.Second, ClaimTTL: 5 * time.Second,
		Worker: WorkerFunc(func(ctx context.Context, _ Request) (Response, error) {
			close(started)
			select {
			case <-ctx.Done():
				return Response{}, ctx.Err()
			case <-time.After(40 * time.Millisecond):
				return Response{SchemaVersion: SchemaVersion, DocumentID: docID, ProcedureVersionID: ver, Outcome: OutcomeReady}, nil
			}
		}),
	}
	parent, cancel := context.WithCancel(context.Background())
	go func() {
		<-started
		cancel()
	}()
	got, err := svc.Request(parent, docID, ver, uuid.New(), uuid.New(), false)
	if err != nil {
		t.Fatal(err)
	}
	if got.ErrorCode != "" || store.finish.Outcome != OutcomeReady {
		t.Fatalf("%+v finish %+v", got, store.finish)
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

func TestWorkerTransportErrorStaysWorkerFailed(t *testing.T) {
	docID := uuid.New()
	ver := uuid.New()
	store := &fakeStore{claim: ClaimResult{JobID: uuid.New(), ClaimToken: uuid.New(), RunWorker: true}}
	svc := &Service{
		XAID: "xa_chu_se", Store: store, Timeout: time.Second, ClaimTTL: 5 * time.Second,
		Worker: WorkerFunc(func(context.Context, Request) (Response, error) {
			return Response{}, errors.New("index worker status 422")
		}),
	}
	got, err := svc.Request(context.Background(), docID, ver, uuid.New(), uuid.New(), false)
	if err != nil {
		t.Fatal(err)
	}
	if got.ErrorCode != "worker_failed" || store.finish.Outcome != OutcomeFailed {
		t.Fatalf("%+v finish %+v", got, store.finish)
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

func TestReindexClaimsRangeWithoutMutatingLink(t *testing.T) {
	docID := uuid.New()
	ver := uuid.New()
	store := &fakeStore{claim: ClaimResult{LinkStatus: "READY", DocumentStatus: "READY"}}
	svc := &Service{XAID: "xa_chu_se", Store: store, Timeout: time.Second, ClaimTTL: time.Second}
	page := "10-21"
	if _, err := svc.Reindex(context.Background(), docID, ver, uuid.New(), uuid.New(), &page); err != nil {
		t.Fatal(err)
	}
	if store.lastClaim.RangeMode != RangeSet || store.lastClaim.RangeValue != "10-21" {
		t.Fatalf("claim %+v", store.lastClaim)
	}
	if store.finish.Outcome != "" {
		t.Fatalf("finish ran on replay-style claim: %+v", store.finish)
	}
}

func TestReindexKeepAllAndSetUseDistinctHashes(t *testing.T) {
	docID := uuid.New()
	ver := uuid.New()
	store := &fakeStore{claim: ClaimResult{LinkStatus: "READY", DocumentStatus: "READY"}}
	svc := &Service{XAID: "xa_chu_se", Store: store, Timeout: time.Second, ClaimTTL: time.Second}
	if _, err := svc.Reindex(context.Background(), docID, ver, uuid.New(), uuid.New(), nil); err != nil {
		t.Fatal(err)
	}
	keep := store.lastClaim.PayloadHash
	empty := ""
	if _, err := svc.Reindex(context.Background(), docID, ver, uuid.New(), uuid.New(), &empty); err != nil {
		t.Fatal(err)
	}
	all := store.lastClaim.PayloadHash
	allWord := "all"
	if _, err := svc.Reindex(context.Background(), docID, ver, uuid.New(), uuid.New(), &allWord); err != nil {
		t.Fatal(err)
	}
	if store.lastClaim.PayloadHash != all || store.lastClaim.RangeMode != RangeAll {
		t.Fatalf("all %+v", store.lastClaim)
	}
	set := "10-21"
	if _, err := svc.Reindex(context.Background(), docID, ver, uuid.New(), uuid.New(), &set); err != nil {
		t.Fatal(err)
	}
	setHash := store.lastClaim.PayloadHash
	if keep == all || keep == setHash || all == setHash {
		t.Fatalf("hashes keep=%s all=%s set=%s", keep, all, setHash)
	}
}

func TestFailedReindexDoesNotFinishReady(t *testing.T) {
	docID := uuid.New()
	ver := uuid.New()
	store := &fakeStore{claim: ClaimResult{
		JobID: uuid.New(), ClaimToken: uuid.New(), RunWorker: true,
		WorkerRequest: Request{SchemaVersion: SchemaVersion, DocumentID: docID, ProcedureVersionID: ver},
	}}
	svc := &Service{
		XAID: "xa_chu_se", Store: store, Timeout: time.Second, ClaimTTL: time.Second,
		Worker: WorkerFunc(func(context.Context, Request) (Response, error) {
			code := "worker_failed"
			return Response{SchemaVersion: SchemaVersion, DocumentID: docID, ProcedureVersionID: ver, Outcome: OutcomeFailed, ErrorCode: &code}, nil
		}),
	}
	page := "5-6"
	got, err := svc.Reindex(context.Background(), docID, ver, uuid.New(), uuid.New(), &page)
	if err != nil {
		t.Fatal(err)
	}
	if store.lastClaim.RangeMode != RangeSet || store.lastClaim.RangeValue != "5-6" {
		t.Fatalf("claim %+v", store.lastClaim)
	}
	if got.ErrorCode == "" || store.finish.Outcome != OutcomeFailed {
		t.Fatalf("%+v finish %+v", got, store.finish)
	}
}

func TestReindexRejectsReversedPageRange(t *testing.T) {
	svc := &Service{XAID: "xa_chu_se", Store: &fakeStore{}, Timeout: time.Second, ClaimTTL: time.Second}
	bad := "21-10"
	if _, err := svc.Reindex(context.Background(), uuid.New(), uuid.New(), uuid.New(), uuid.New(), &bad); err == nil {
		t.Fatal("reversed range must fail before claim")
	}
}

func TestPageRangeBounds(t *testing.T) {
	_, _, unconstrained, ok := PageRangeBounds(nil)
	if !ok || !unconstrained {
		t.Fatal("nil should be unconstrained")
	}
	empty := ""
	_, _, unconstrained, ok = PageRangeBounds(&empty)
	if !ok || !unconstrained {
		t.Fatal("empty should be unconstrained")
	}
	page := "10-21"
	start, end, unconstrained, ok := PageRangeBounds(&page)
	if !ok || unconstrained || start != 10 || end != 21 {
		t.Fatalf("10-21 -> %d %d %v %v", start, end, unconstrained, ok)
	}
	bad := "21-10"
	_, _, _, ok = PageRangeBounds(&bad)
	if ok {
		t.Fatal("reversed range must not parse")
	}
}

func TestParseReindexPageRange(t *testing.T) {
	keep, _, key, err := parseReindexPageRange(nil)
	if err != nil || keep != RangeKeep || key != RangeKeep {
		t.Fatalf("keep %s %s %v", keep, key, err)
	}
	empty := "  "
	mode, _, key, err := parseReindexPageRange(&empty)
	if err != nil || mode != RangeAll || key != RangeAll {
		t.Fatalf("empty %s %s %v", mode, key, err)
	}
	page := "10-21"
	mode, value, key, err := parseReindexPageRange(&page)
	if err != nil || mode != RangeSet || value != "10-21" || key != "10-21" {
		t.Fatalf("set %s %s %s %v", mode, value, key, err)
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
