package index

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
)

// leaseStore adds ExtendClaim to fakeStore and records finishes from background goroutines.
type leaseStore struct {
	fakeStore
	mu       sync.Mutex
	extends  int
	alive    atomic.Bool
	finishes []FinishInput
}

func (s *leaseStore) ExtendClaim(_ context.Context, xaID string, jobID, token uuid.UUID, ttl time.Duration) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.extends++
	return s.alive.Load(), nil
}

func (s *leaseStore) Finish(_ context.Context, in FinishInput) (FinishResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.finishes = append(s.finishes, in)
	return FinishResult{Applied: true, LinkStatus: in.Outcome, DocumentStatus: in.Outcome}, nil
}

func newLeaseStore(docID, ver uuid.UUID) *leaseStore {
	s := &leaseStore{fakeStore: fakeStore{claim: ClaimResult{
		JobID: uuid.New(), ClaimToken: uuid.New(), RunWorker: true,
		WorkerRequest: Request{SchemaVersion: SchemaVersion, DocumentID: docID, ProcedureVersionID: ver},
	}}}
	s.alive.Store(true)
	return s
}

func readyWorker(docID, ver uuid.UUID, hold <-chan struct{}) WorkerFunc {
	return func(ctx context.Context, _ Request) (Response, error) {
		select {
		case <-hold:
		case <-ctx.Done():
			return Response{}, ctx.Err()
		}
		return Response{SchemaVersion: SchemaVersion, DocumentID: docID, ProcedureVersionID: ver, Outcome: OutcomeReady}, nil
	}
}

func TestAsyncRequestReturnsClaimedJobBeforeWorkerFinishes(t *testing.T) {
	docID, ver := uuid.New(), uuid.New()
	store := newLeaseStore(docID, ver)
	hold := make(chan struct{})
	svc := &Service{
		XAID: "xa_chu_se", Store: store, Timeout: 5 * time.Second, ClaimTTL: time.Second,
		Async: true, Heartbeat: 10 * time.Millisecond, MaxConcurrent: 1,
		Worker: readyWorker(docID, ver, hold),
	}
	got, err := svc.Request(context.Background(), docID, ver, uuid.New(), uuid.New(), false)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Accepted || got.JobStatus != "CLAIMED" || got.LinkStatus != "PROCESSING" || got.JobID != store.claim.JobID {
		t.Fatalf("accepted result %+v", got)
	}
	time.Sleep(60 * time.Millisecond)
	store.mu.Lock()
	beats, finished := store.extends, len(store.finishes)
	store.mu.Unlock()
	if finished != 0 {
		t.Fatal("finish must wait for the worker")
	}
	if beats < 2 {
		t.Fatalf("heartbeat must renew the lease while the worker runs, got %d", beats)
	}
	close(hold)
	svc.Wait()
	if len(store.finishes) != 1 || store.finishes[0].Outcome != OutcomeReady || store.finishes[0].ClaimToken != store.claim.ClaimToken {
		t.Fatalf("finishes %+v", store.finishes)
	}
}

func TestAsyncLostClaimCancelsWorkerAndFinishesClaimLost(t *testing.T) {
	docID, ver := uuid.New(), uuid.New()
	store := newLeaseStore(docID, ver)
	store.alive.Store(false)
	var cancelled atomic.Bool
	svc := &Service{
		XAID: "xa_chu_se", Store: store, Timeout: 5 * time.Second, ClaimTTL: time.Second,
		Async: true, Heartbeat: 10 * time.Millisecond, MaxConcurrent: 1,
		Worker: WorkerFunc(func(ctx context.Context, _ Request) (Response, error) {
			<-ctx.Done()
			cancelled.Store(true)
			return Response{}, ctx.Err()
		}),
	}
	if _, err := svc.Request(context.Background(), docID, ver, uuid.New(), uuid.New(), true); err != nil {
		t.Fatal(err)
	}
	svc.Wait()
	if !cancelled.Load() {
		t.Fatal("losing the claim must cancel the worker call")
	}
	if len(store.finishes) != 1 || store.finishes[0].Outcome != OutcomeFailed || store.finishes[0].ErrorCode != "claim_lost" {
		t.Fatalf("finishes %+v", store.finishes)
	}
}

func TestAsyncMaxConcurrentRunsJobsOneAtATime(t *testing.T) {
	docID, ver := uuid.New(), uuid.New()
	store := newLeaseStore(docID, ver)
	var running, peak atomic.Int32
	svc := &Service{
		XAID: "xa_chu_se", Store: store, Timeout: 5 * time.Second, ClaimTTL: time.Second,
		Async: true, Heartbeat: 10 * time.Millisecond, MaxConcurrent: 1,
		Worker: WorkerFunc(func(ctx context.Context, _ Request) (Response, error) {
			n := running.Add(1)
			for {
				p := peak.Load()
				if n <= p || peak.CompareAndSwap(p, n) {
					break
				}
			}
			time.Sleep(30 * time.Millisecond)
			running.Add(-1)
			return Response{SchemaVersion: SchemaVersion, DocumentID: docID, ProcedureVersionID: ver, Outcome: OutcomeReady}, nil
		}),
	}
	for i := 0; i < 3; i++ {
		if _, err := svc.Request(context.Background(), docID, ver, uuid.New(), uuid.New(), false); err != nil {
			t.Fatal(err)
		}
	}
	svc.Wait()
	if peak.Load() != 1 {
		t.Fatalf("MaxConcurrent=1 but %d workers overlapped", peak.Load())
	}
	if len(store.finishes) != 3 {
		t.Fatalf("finishes %d", len(store.finishes))
	}
}

func TestAsyncReindexKeepsLinkReady(t *testing.T) {
	docID, ver := uuid.New(), uuid.New()
	store := newLeaseStore(docID, ver)
	svc := &Service{
		XAID: "xa_chu_se", Store: store, Timeout: time.Second, ClaimTTL: time.Second,
		Async: true, Heartbeat: 10 * time.Millisecond, MaxConcurrent: 1,
		Worker: readyWorker(docID, ver, closedChan()),
	}
	got, err := svc.Reindex(context.Background(), docID, ver, uuid.New(), uuid.New(), nil)
	if err != nil {
		t.Fatal(err)
	}
	svc.Wait()
	if !got.Accepted || got.LinkStatus != "READY" {
		t.Fatalf("reindex result %+v", got)
	}
}

func TestSyncModeIsUnchanged(t *testing.T) {
	docID, ver := uuid.New(), uuid.New()
	store := newLeaseStore(docID, ver)
	svc := &Service{
		XAID: "xa_chu_se", Store: store, Timeout: time.Second, ClaimTTL: time.Second,
		Worker: readyWorker(docID, ver, closedChan()),
	}
	got, err := svc.Request(context.Background(), docID, ver, uuid.New(), uuid.New(), false)
	if err != nil {
		t.Fatal(err)
	}
	if got.Accepted || got.LinkStatus != OutcomeReady || len(store.finishes) != 1 {
		t.Fatalf("sync result %+v finishes %d", got, len(store.finishes))
	}
	if store.extends != 0 {
		t.Fatal("sync mode must not heartbeat")
	}
}

func closedChan() <-chan struct{} {
	ch := make(chan struct{})
	close(ch)
	return ch
}
