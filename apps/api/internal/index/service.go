package index

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"log"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
)

var pageRangePattern = regexp.MustCompile(`^[1-9][0-9]{0,3}(-[1-9][0-9]{0,3})?$`)

type LinkInput struct {
	DocumentID         uuid.UUID
	ProcedureVersionID uuid.UUID
	RelationshipType   string
	PageRange          string
	ActorUserID        uuid.UUID
	RequestID          uuid.UUID
}

type Link struct {
	DocumentID         uuid.UUID  `json:"document_id"`
	ProcedureID        uuid.UUID  `json:"procedure_id"`
	ProcedureVersionID uuid.UUID  `json:"procedure_version_id"`
	ProcedureCode      string     `json:"procedure_code"`
	Version            string     `json:"version"`
	RelationshipType   string     `json:"relationship_type"`
	PageRange          *string    `json:"page_range,omitempty"`
	IndexStatus        string     `json:"index_status"`
	LastErrorCode      *string    `json:"last_error_code,omitempty"`
	UpdatedAt          time.Time  `json:"updated_at"`
	ClaimExpiresAt     *time.Time `json:"claim_expires_at,omitempty"`
	Recoverable        bool       `json:"recoverable"`
	ReindexErrorCode   *string    `json:"reindex_error_code,omitempty"`
	PageCount          *int       `json:"page_count,omitempty"`
	NativePageCount    *int       `json:"native_page_count,omitempty"`
	OCRPageCount       *int       `json:"ocr_page_count,omitempty"`
	ChunkCount         *int       `json:"chunk_count,omitempty"`
	PipelineVersion    *string    `json:"pipeline_version,omitempty"`
	EmbeddingModelID   *string    `json:"embedding_model_id,omitempty"`
}

type Target struct {
	ProcedureID        uuid.UUID `json:"procedure_id"`
	ProcedureCode      string    `json:"procedure_code"`
	ProcedureName      string    `json:"procedure_name"`
	DomainID           string    `json:"domain_id"`
	ProcedureVersionID uuid.UUID `json:"procedure_version_id"`
	Version            string    `json:"version"`
	Status             string    `json:"status"`
}

type Store interface {
	ListTargets(ctx context.Context, xaID string) ([]Target, error)
	ListLinks(ctx context.Context, xaID string, documentID uuid.UUID) ([]Link, error)
	Link(ctx context.Context, xaID string, in LinkInput, payloadHash string) (Link, bool, error)
	Unlink(ctx context.Context, xaID string, documentID, versionID, actor, requestID uuid.UUID, payloadHash string) (bool, error)
	Claim(ctx context.Context, in ClaimInput) (ClaimResult, error)
	Finish(ctx context.Context, in FinishInput) (FinishResult, error)
	StageGeneration(ctx context.Context, in StageInput) (StagedObject, error)
}

type ClaimInput struct {
	DocumentID         uuid.UUID
	ProcedureVersionID uuid.UUID
	XAID               string
	RequestID          uuid.UUID
	ActorUserID        uuid.UUID
	PayloadHash        string
	Retry              bool
	Reindex            bool
	AuditAction        string
	TTL                time.Duration
}

type ClaimResult struct {
	JobID          uuid.UUID
	ClaimToken     uuid.UUID
	RunWorker      bool
	LinkStatus     string
	DocumentStatus string
	ErrorCode      string
	WorkerRequest  Request
}

type FinishInput struct {
	JobID              uuid.UUID
	ClaimToken         uuid.UUID
	DocumentID         uuid.UUID
	ProcedureVersionID uuid.UUID
	XAID               string
	ActorUserID        uuid.UUID
	RequestID          uuid.UUID
	Outcome            string
	ErrorCode          string
	GenerationID       uuid.UUID
	ManifestHash       string
	ChunkCount         int
	VectorCount        int
}

type StageInput struct {
	DocumentID         uuid.UUID
	ProcedureVersionID uuid.UUID
	JobID              uuid.UUID
	XAID               string
	Bucket             string
}

type StagedObject struct {
	GenerationID uuid.UUID
	ProcedureID  uuid.UUID
	Bucket       string
	ObjectKey    string
	Checksum     string
}

type FinishResult struct {
	Applied        bool
	LinkStatus     string
	DocumentStatus string
}

type GenerationMetrics struct {
	Ready        int `json:"ready"`
	Staging      int `json:"staging"`
	Failed       int `json:"failed"`
	Orphan       int `json:"orphan"`
	RetryCount   int `json:"retry_count"`
	ReindexCount int `json:"reindex_count"`
}

type Service struct {
	XAID           string
	Store          Store
	Worker         Worker
	Timeout        time.Duration
	ClaimTTL       time.Duration
	Pipeline       bool
	Bucket         string
	PublishTimeout time.Duration
}

func (s *Service) Metrics(ctx context.Context) (GenerationMetrics, error) {
	repo, ok := s.Store.(interface {
		GenerationMetrics(context.Context, string) (GenerationMetrics, error)
	})
	if !ok {
		return GenerationMetrics{}, nil
	}
	return repo.GenerationMetrics(ctx, s.XAID)
}

func (s *Service) ListTargets(ctx context.Context) ([]Target, error) {
	items, err := s.Store.ListTargets(ctx, s.XAID)
	if items == nil {
		items = []Target{}
	}
	return items, err
}

func (s *Service) ListLinks(ctx context.Context, documentID uuid.UUID) ([]Link, error) {
	items, err := s.Store.ListLinks(ctx, s.XAID, documentID)
	if errors.Is(err, ErrNotFound) {
		return nil, ErrNotFound
	}
	if items == nil {
		items = []Link{}
	}
	return items, err
}

func (s *Service) Link(ctx context.Context, in LinkInput) (Link, bool, error) {
	rel, err := cleanRelationship(in.RelationshipType)
	if err != nil {
		return Link{}, false, err
	}
	page, err := cleanPageRange(in.PageRange)
	if err != nil {
		return Link{}, false, err
	}
	in.RelationshipType = rel
	in.PageRange = page
	return s.Store.Link(ctx, s.XAID, in, hashPayload("LINK", in.DocumentID, in.ProcedureVersionID, rel, page))
}

func (s *Service) Unlink(ctx context.Context, documentID, versionID, actor, requestID uuid.UUID) (bool, error) {
	return s.Store.Unlink(ctx, s.XAID, documentID, versionID, actor, requestID, hashPayload("UNLINK", documentID, versionID, "", ""))
}

func (s *Service) Request(ctx context.Context, documentID, versionID, actor, requestID uuid.UUID, retry bool) (Result, error) {
	return s.dispatch(ctx, documentID, versionID, actor, requestID, retry, false)
}

func (s *Service) Reindex(ctx context.Context, documentID, versionID, actor, requestID uuid.UUID) (Result, error) {
	return s.dispatch(ctx, documentID, versionID, actor, requestID, false, true)
}

func (s *Service) dispatch(ctx context.Context, documentID, versionID, actor, requestID uuid.UUID, retry, reindex bool) (Result, error) {
	kind := "index"
	action := "DOCUMENT_INDEX_REQUESTED"
	if retry {
		kind = "retry"
		action = "DOCUMENT_INDEX_RETRY"
	}
	if reindex {
		kind = "reindex"
		action = "DOCUMENT_REINDEX_REQUESTED"
	}
	claim, err := s.Store.Claim(ctx, ClaimInput{
		DocumentID: documentID, ProcedureVersionID: versionID, XAID: s.XAID,
		RequestID: requestID, ActorUserID: actor,
		PayloadHash: hashPayload(kind, documentID, versionID, "", ""),
		Retry:       retry, Reindex: reindex, AuditAction: action, TTL: s.ClaimTTL,
	})
	if err != nil {
		return Result{}, err
	}
	if !claim.RunWorker {
		return Result{
			DocumentID: documentID, ProcedureVersionID: versionID,
			LinkStatus: claim.LinkStatus, ProcessingStatus: claim.DocumentStatus,
			JobID: claim.JobID, ErrorCode: claim.ErrorCode, Replay: true,
		}, nil
	}
	var staged StagedObject
	if s.Pipeline {
		staged, err = s.Store.StageGeneration(ctx, StageInput{
			DocumentID: documentID, ProcedureVersionID: versionID, JobID: claim.JobID,
			XAID: s.XAID, Bucket: s.Bucket,
		})
		if err != nil {
			_, _ = s.finishDetached(claim, documentID, versionID, actor, requestID, OutcomeFailed, "postgres_failed", uuid.Nil, "", 0, 0)
			return Result{}, err
		}
		claim.WorkerRequest.SchemaVersion = SchemaVersionV2
		claim.WorkerRequest.ProcedureID = staged.ProcedureID
		claim.WorkerRequest.JobID = claim.JobID
		claim.WorkerRequest.ClaimToken = claim.ClaimToken
		claim.WorkerRequest.GenerationID = staged.GenerationID
		claim.WorkerRequest.Bucket = staged.Bucket
		claim.WorkerRequest.ObjectKey = staged.ObjectKey
		claim.WorkerRequest.PipelineVersion = PipelineVersion
	}
	timeout := s.Timeout
	if timeout <= 0 {
		timeout = 3 * time.Second
	}
	wctx, cancel := context.WithTimeout(ctx, timeout)
	resp, callErr := s.Worker.Index(wctx, claim.WorkerRequest)
	cancel()
	code := ""
	outcome := OutcomeReady
	if callErr != nil || resp.Outcome != OutcomeReady || resp.DocumentID != documentID || resp.ProcedureVersionID != versionID {
		outcome = OutcomeFailed
		code = "worker_failed"
		if errors.Is(callErr, context.DeadlineExceeded) || errors.Is(wctx.Err(), context.DeadlineExceeded) {
			code = "timeout"
		} else if resp.DocumentID != documentID || resp.ProcedureVersionID != versionID {
			code = "worker_mismatch"
		} else if resp.ErrorCode != nil && *resp.ErrorCode != "" && len(*resp.ErrorCode) <= MaxErrorCodeLen {
			code = *resp.ErrorCode
		}
	}
	if s.Pipeline && outcome == OutcomeReady {
		if resp.SchemaVersion != SchemaVersionV2 || resp.GenerationID != staged.GenerationID || resp.JobID != claim.JobID || resp.XAID != s.XAID || resp.ChunkCount < 1 || resp.ChunkCount != resp.VectorCount || resp.ManifestHash == "" {
			outcome = OutcomeFailed
			code = "publish_mismatch"
		}
	}
	if s.Pipeline && resp.GenerationID != uuid.Nil && resp.GenerationID != staged.GenerationID {
		outcome = OutcomeFailed
		code = "worker_mismatch"
	}
	genID := staged.GenerationID
	finished, err := s.finishDetached(claim, documentID, versionID, actor, requestID, outcome, code, genID, resp.ManifestHash, resp.ChunkCount, resp.VectorCount)
	if err != nil {
		return Result{}, err
	}
	if !finished.Applied {
		return Result{}, ErrConflict
	}
	log.Printf("index_finish outcome=%s error_code=%s chunks=%d vectors=%d", outcome, code, resp.ChunkCount, resp.VectorCount)
	return Result{
		DocumentID: documentID, ProcedureVersionID: versionID,
		LinkStatus: finished.LinkStatus, ProcessingStatus: finished.DocumentStatus,
		JobID: claim.JobID, ErrorCode: code,
	}, nil
}

func (s *Service) finishDetached(claim ClaimResult, documentID, versionID, actor, requestID uuid.UUID, outcome, code string, generationID uuid.UUID, manifest string, chunks, vectors int) (FinishResult, error) {
	timeout := s.PublishTimeout
	if timeout <= 0 {
		timeout = 10 * time.Second
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return s.Store.Finish(ctx, FinishInput{
		JobID: claim.JobID, ClaimToken: claim.ClaimToken,
		DocumentID: documentID, ProcedureVersionID: versionID, XAID: s.XAID,
		ActorUserID: actor, RequestID: requestID, Outcome: outcome, ErrorCode: code,
		GenerationID: generationID, ManifestHash: manifest, ChunkCount: chunks, VectorCount: vectors,
	})
}

func hashPayload(kind string, documentID, versionID uuid.UUID, rel, page string) string {
	sum := sha256.Sum256([]byte(kind + "\n" + documentID.String() + "\n" + versionID.String() + "\n" + rel + "\n" + page))
	return hex.EncodeToString(sum[:])
}

func cleanRelationship(v string) (string, error) {
	v = strings.TrimSpace(v)
	if v != "SOURCE" && v != "SUPERSEDES" {
		return "", ErrValidation
	}
	return v, nil
}

func cleanPageRange(v string) (string, error) {
	v = strings.TrimSpace(v)
	if v == "" {
		return "", nil
	}
	if !pageRangePattern.MatchString(v) {
		return "", ErrValidation
	}
	dash := strings.IndexByte(v, '-')
	if dash < 0 {
		return v, nil
	}
	start, errStart := strconv.Atoi(v[:dash])
	end, errEnd := strconv.Atoi(v[dash+1:])
	if errStart != nil || errEnd != nil || start > end {
		return "", ErrValidation
	}
	return v, nil
}

func MarkRecoverable(link *Link, now time.Time) {
	expired := link.ClaimExpiresAt == nil || !link.ClaimExpiresAt.After(now)
	link.Recoverable = link.IndexStatus == "FAILED" || (link.IndexStatus == "PROCESSING" && expired)
}
