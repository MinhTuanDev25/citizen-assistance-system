package activation

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"
)

const (
	ActionActivate        = "ACTIVATE_VERSION"
	ActionCreate          = "CREATE_VERSION"
	ActionCreateProcedure = "CREATE_PROCEDURE"
	AuditAction           = "PROCEDURE_VERSION_ACTIVATED"
	AuditCreate           = "PROCEDURE_VERSION_CREATED"
	AuditCreateProcedure  = "PROCEDURE_CREATED"

	// StatusSource marks the version a new procedure gets for linking and
	// indexing its first PDF. Empty definition; never activatable.
	StatusSource       = "SOURCE"
	SourceVersionLabel = "0.0.0"

	RequiredVectorDim = 384
)

type ProcedureSummary struct {
	ID              uuid.UUID  `json:"id"`
	ProcedureCode   string     `json:"procedure_code"`
	Name            string     `json:"name"`
	DomainID        string     `json:"domain_id"`
	DomainName      string     `json:"domain_name"`
	XaID            string     `json:"xa_id"`
	ActiveVersionID *uuid.UUID `json:"active_version_id"`
	ActiveVersion   string     `json:"active_version,omitempty"`
	VersionCount    int        `json:"version_count"`
	LegacyActive    bool       `json:"legacy_active"`
}

type VersionSummary struct {
	ID           uuid.UUID       `json:"id"`
	ProcedureID  uuid.UUID       `json:"procedure_id"`
	Version      string          `json:"version"`
	Status       string          `json:"status"`
	CreatedAt    time.Time       `json:"created_at"`
	CreatedBy    uuid.UUID       `json:"created_by"`
	ApprovedBy   *uuid.UUID      `json:"approved_by,omitempty"`
	ApprovedAt   *time.Time      `json:"approved_at,omitempty"`
	ActivatedAt  *time.Time      `json:"activated_at,omitempty"`
	ActivatedBy  *uuid.UUID      `json:"activated_by,omitempty"`
	Definition   json.RawMessage `json:"definition,omitempty"`
	Documents    []DocumentLink  `json:"documents"`
	Eligibility  Eligibility     `json:"eligibility"`
	LegacyActive bool            `json:"legacy_active"`
}

type DocumentLink struct {
	DocumentID         uuid.UUID  `json:"document_id"`
	Title              string     `json:"title"`
	RelationshipType   string     `json:"relationship_type"`
	IndexStatus        string     `json:"index_status"`
	ValidityStatus     string     `json:"validity_status"`
	ActiveGenerationID *uuid.UUID `json:"active_generation_id,omitempty"`
	GenerationStatus   string     `json:"generation_status,omitempty"`
	ChunkCount         int        `json:"chunk_count,omitempty"`
	VectorCount        int        `json:"vector_count,omitempty"`
	VectorDimension    int        `json:"vector_dimension,omitempty"`
	EmbeddingModelID   string     `json:"embedding_model_id,omitempty"`
	PublishedAt        *time.Time `json:"published_at,omitempty"`
	Reasons            []string   `json:"reasons,omitempty"`
}

type Eligibility struct {
	CanActivate bool     `json:"can_activate"`
	Reasons     []string `json:"reasons"`
}

type ActivateRequest struct {
	ExpectedActiveVersionID *uuid.UUID `json:"expected_active_version_id"`
}

type ActivateResult struct {
	ProcedureID             uuid.UUID  `json:"procedure_id"`
	PreviousActiveVersionID *uuid.UUID `json:"previous_active_version_id"`
	ActiveVersionID         uuid.UUID  `json:"active_version_id"`
	Version                 string     `json:"version"`
	Status                  string     `json:"status"`
	ActivatedAt             time.Time  `json:"activated_at"`
	ActivatedBy             uuid.UUID  `json:"activated_by"`
	Replayed                bool       `json:"replayed"`
	Noop                    bool       `json:"noop,omitempty"`
}

type CreateVersionRequest struct {
	Version           string          `json:"version"`
	CopyFromVersionID *uuid.UUID      `json:"copy_from_version_id"`
	SourceDraftID     *uuid.UUID      `json:"source_draft_id,omitempty"`
	Definition        json.RawMessage `json:"definition"`
	Confirm           bool            `json:"confirm"`
}

type CreateProcedureRequest struct {
	ProcedureCode string `json:"procedure_code"`
	Name          string `json:"name"`
	DomainID      string `json:"domain_id"`
}

type CreateProcedureResult struct {
	ID              uuid.UUID `json:"id"`
	ProcedureCode   string    `json:"procedure_code"`
	Name            string    `json:"name"`
	DomainID        string    `json:"domain_id"`
	XaID            string    `json:"xa_id"`
	SourceVersionID uuid.UUID `json:"source_version_id"`
	SourceVersion   string    `json:"source_version"`
	Replayed        bool      `json:"replayed"`
}

type CreateVersionResult struct {
	ID          uuid.UUID `json:"id"`
	ProcedureID uuid.UUID `json:"procedure_id"`
	Version     string    `json:"version"`
	Status      string    `json:"status"`
	Replayed    bool      `json:"replayed"`
}

type HistoryItem struct {
	RequestID               uuid.UUID  `json:"request_id"`
	ActorUserID             *uuid.UUID `json:"actor_user_id,omitempty"`
	ProcedureID             uuid.UUID  `json:"procedure_id"`
	PreviousActiveVersionID *uuid.UUID `json:"previous_active_version_id,omitempty"`
	NewActiveVersionID      uuid.UUID  `json:"new_active_version_id"`
	CreatedAt               time.Time  `json:"created_at"`
}

type versionRow struct {
	ID            uuid.UUID
	ProcedureID   uuid.UUID
	Version       string
	Status        string
	Definition    json.RawMessage
	CreatedBy     uuid.UUID
	ApprovedBy    *uuid.UUID
	CreatedAt     time.Time
	ApprovedAt    *time.Time
	ActivatedAt   *time.Time
	ActivatedBy   *uuid.UUID
	XaID          string
	DomainID      string
	ProcedureCode string
	ProcedureName string
}

type sourceRow struct {
	DocumentID         uuid.UUID
	Title              string
	RelationshipType   string
	IndexStatus        string
	ValidityStatus     string
	Unlinked           bool
	XaID               string
	DomainID           string
	ProcedureID        uuid.UUID
	ProcedureVersionID uuid.UUID
	ActiveGenerationID *uuid.UUID
	GenerationStatus   string
	ChunkCount         int
	VectorCount        int
	VectorDimension    int
	EmbeddingModelID   string
	PublishedAt        *time.Time
}
