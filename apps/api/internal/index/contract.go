package index

import (
	"context"
	"errors"

	"github.com/google/uuid"
)

const SchemaVersion = "index.v1"

// SchemaVersionV2 is the offline content pipeline. Mock workers stay on v1.
const SchemaVersionV2 = "index.v2"

const PipelineVersion = "p4b.1"

// MaxErrorCodeLen is the stored and accepted length of an indexing error code.
const MaxErrorCodeLen = 64

// MaxIndexBodyBytes is the accepted size of POST /v1/index.
const MaxIndexBodyBytes = 8 << 10

const (
	OutcomeReady  = "READY"
	OutcomeFailed = "FAILED"
)

var (
	ErrNotFound     = errors.New("not found")
	ErrConflict     = errors.New("conflict")
	ErrInvalidState = errors.New("invalid state")
	ErrNotLinked    = errors.New("not linked")
	ErrDuplicate    = errors.New("duplicate")
	ErrIdempotency  = errors.New("idempotency conflict")
	ErrValidation   = errors.New("validation")
)

// Request is the strict indexing contract. It carries identifiers only.
// PDF bytes, storage URIs, and embeddings are not part of P4A.
type Request struct {
	SchemaVersion      string    `json:"schema_version"`
	DocumentID         uuid.UUID `json:"document_id"`
	XAID               string    `json:"xa_id"`
	DomainID           string    `json:"domain_id"`
	Checksum           string    `json:"checksum"`
	ProcedureVersionID uuid.UUID `json:"procedure_version_id"`
	RelationshipType   string    `json:"relationship_type"`
	PageRange          *string   `json:"page_range"`
	ProcedureID        uuid.UUID `json:"procedure_id,omitempty"`
	JobID              uuid.UUID `json:"job_id,omitempty"`
	ClaimToken         uuid.UUID `json:"claim_token,omitempty"`
	GenerationID       uuid.UUID `json:"generation_id,omitempty"`
	Bucket             string    `json:"bucket,omitempty"`
	ObjectKey          string    `json:"object_key,omitempty"`
	PipelineVersion    string    `json:"pipeline_version,omitempty"`
}

type Response struct {
	SchemaVersion      string    `json:"schema_version"`
	DocumentID         uuid.UUID `json:"document_id"`
	ProcedureVersionID uuid.UUID `json:"procedure_version_id"`
	Outcome            string    `json:"outcome"`
	ErrorCode          *string   `json:"error_code"`
	XAID               string    `json:"xa_id,omitempty"`
	JobID              uuid.UUID `json:"job_id,omitempty"`
	GenerationID       uuid.UUID `json:"generation_id,omitempty"`
	SourceSHA256       string    `json:"source_sha256,omitempty"`
	ContentSHA256      string    `json:"content_sha256,omitempty"`
	PagesProcessed     int       `json:"pages_processed,omitempty"`
	NativePages        int       `json:"native_pages,omitempty"`
	OCRPages           int       `json:"ocr_pages,omitempty"`
	ChunkCount         int       `json:"chunk_count,omitempty"`
	VectorCount        int       `json:"vector_count,omitempty"`
	ManifestHash       string    `json:"manifest_hash,omitempty"`
	PipelineVersion    string    `json:"pipeline_version,omitempty"`
	ExtractionVersion  string    `json:"extraction_version,omitempty"`
	OCRVersion         string    `json:"ocr_version,omitempty"`
	EmbeddingModelID   string    `json:"embedding_model_id,omitempty"`
	EmbeddingRevision  string    `json:"embedding_revision,omitempty"`
	EmbeddingChecksum  string    `json:"embedding_checksum,omitempty"`
	VectorDimension    int       `json:"vector_dimension,omitempty"`
}

// Worker indexes one attached document. P4A supplies a mock, not an embedder.
type Worker interface {
	Index(ctx context.Context, req Request) (Response, error)
}

type Result struct {
	DocumentID         uuid.UUID `json:"document_id"`
	ProcedureVersionID uuid.UUID `json:"procedure_version_id"`
	LinkStatus         string    `json:"link_status"`
	ProcessingStatus   string    `json:"processing_status"`
	JobID              uuid.UUID `json:"job_id,omitempty"`
	ErrorCode          string    `json:"error_code,omitempty"`
	Replay             bool      `json:"idempotent_replay,omitempty"`
}
