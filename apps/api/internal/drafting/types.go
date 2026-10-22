package drafting

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
)

type Excerpt struct {
	ExcerptID    uuid.UUID `json:"excerpt_id"`
	DocumentID   uuid.UUID `json:"document_id"`
	GenerationID uuid.UUID `json:"generation_id"`
	PageStart    int       `json:"page_start"`
	PageEnd      int       `json:"page_end"`
	TextSHA256   string    `json:"text_sha256"`
	Content      string    `json:"content"`
	ChunkIndex   int       `json:"-"`
}

type SourceSnapshot struct {
	ProcedureID         uuid.UUID
	ProcedureCode       string
	ProcedureName       string
	XaID                string
	DocumentID          uuid.UUID
	DocumentSHA256      string
	GenerationID        uuid.UUID
	GenerationStatus    string
	GenerationVersionID uuid.UUID
	GenerationVersion   string
	GenerationPVStatus  string
	SourcePageStart     int
	SourcePageEnd       int
	Excerpts            []Excerpt
}

type EvidenceItem struct {
	ProposalID          string `json:"proposal_id"`
	RawProposalID       string `json:"raw_proposal_id,omitempty"`
	Kind                string `json:"kind"`
	ExcerptID           string `json:"excerpt_id"`
	Page                int    `json:"page"`
	Quotation           string `json:"quotation"`
	ReviewWarning       string `json:"review_warning"`
	Status              string `json:"status,omitempty"`
	QuoteValid          bool   `json:"quote_valid"`
	ProposalFingerprint string `json:"proposal_fingerprint,omitempty"`
	BoundFingerprint    string `json:"bound_fingerprint,omitempty"`
	ReviewStatus        string `json:"review_status,omitempty"`
	SourceSHA256        string `json:"source_sha256,omitempty"`
}

type GenerateRequest struct {
	DocumentID      uuid.UUID `json:"document_id"`
	GenerationID    uuid.UUID `json:"generation_id"`
	SourcePageStart int       `json:"source_page_start"`
	SourcePageEnd   int       `json:"source_page_end"`
}

type PatchRequest struct {
	Definition        json.RawMessage `json:"definition"`
	ExpectedUpdatedAt *time.Time      `json:"expected_updated_at"`
}

type ReviewDecision struct {
	ProposalID string `json:"proposal_id"`
	Decision   string `json:"decision"`
}

type ReviewRequest struct {
	ExpectedUpdatedAt *time.Time       `json:"expected_updated_at"`
	Decisions         []ReviewDecision `json:"decisions"`
}

type ApproveRequest struct {
	Confirm           bool       `json:"confirm"`
	ExpectedUpdatedAt *time.Time `json:"expected_updated_at"`
}

type ActivateDraftRequest struct {
	Confirm                 bool       `json:"confirm"`
	ExpectedUpdatedAt       *time.Time `json:"expected_updated_at"`
	ExpectedActiveVersionID *uuid.UUID `json:"expected_active_version_id"`
	Version                 string     `json:"version"`
	PageRange               string     `json:"page_range"`
}

type DraftView struct {
	ID                    uuid.UUID                `json:"id"`
	ProcedureID           uuid.UUID                `json:"procedure_id"`
	ProcedureCode         string                   `json:"procedure_code"`
	DocumentID            uuid.UUID                `json:"document_id"`
	Status                string                   `json:"status"`
	Definition            json.RawMessage          `json:"definition"`
	Evidence              []EvidenceItem           `json:"evidence"`
	Warnings              []string                 `json:"warnings"`
	MissingRequirements   []string                 `json:"missing_requirements"`
	Validation            decision.DraftValidation `json:"validation"`
	Model                 string                   `json:"model"`
	Provider              string                   `json:"provider"`
	PromptVersion         string                   `json:"prompt_version"`
	PromptHash            string                   `json:"prompt_hash"`
	GenerationID          uuid.UUID                `json:"generation_id"`
	SourcePageStart       int                      `json:"source_page_start"`
	SourcePageEnd         int                      `json:"source_page_end"`
	Usage                 map[string]any           `json:"usage,omitempty"`
	LatencyMS             int                      `json:"latency_ms,omitempty"`
	CreatedBy             *uuid.UUID               `json:"created_by,omitempty"`
	UpdatedBy             *uuid.UUID               `json:"updated_by,omitempty"`
	CreatedAt             time.Time                `json:"created_at"`
	UpdatedAt             time.Time                `json:"updated_at"`
	Revision              int                      `json:"revision"`
	Proposals             []MaterialProposal       `json:"proposals,omitempty"`
	HistoricalNote        string                   `json:"historical_generation_note,omitempty"`
	ApprovedBy            *uuid.UUID               `json:"approved_by,omitempty"`
	ApprovedAt            *time.Time               `json:"approved_at,omitempty"`
	PublishedVersionID    *uuid.UUID               `json:"published_version_id,omitempty"`
	PublishedVersion      string                   `json:"published_version,omitempty"`
	IdentityMismatch      bool                     `json:"identity_mismatch,omitempty"`
	OriginalProcedureCode string                   `json:"original_procedure_code,omitempty"`
	IndexRequestID        *uuid.UUID               `json:"index_request_id,omitempty"`
	IndexJobID            *uuid.UUID               `json:"index_job_id,omitempty"`
}

type storedPayload struct {
	SchemaVersion         string          `json:"schema_version"`
	Definition            json.RawMessage `json:"definition"`
	Evidence              []EvidenceItem  `json:"evidence"`
	Warnings              []string        `json:"warnings"`
	MissingRequirements   []string        `json:"missing_requirements"`
	Model                 string          `json:"model"`
	Provider              string          `json:"provider"`
	PromptVersion         string          `json:"prompt_version"`
	PromptHash            string          `json:"prompt_hash"`
	GenerationID          uuid.UUID       `json:"generation_id"`
	DocumentID            uuid.UUID       `json:"document_id"`
	SourcePageStart       int             `json:"source_page_start"`
	SourcePageEnd         int             `json:"source_page_end"`
	Usage                 map[string]any  `json:"usage,omitempty"`
	LatencyMS             int             `json:"latency_ms,omitempty"`
	HistoricalNote        string          `json:"historical_generation_note,omitempty"`
	OriginalProcedureCode string          `json:"original_procedure_code,omitempty"`
	IdentityMismatch      bool            `json:"identity_mismatch,omitempty"`
	Revision              int             `json:"revision"`
	ApprovedBy            *uuid.UUID      `json:"approved_by,omitempty"`
	ApprovedAt            *time.Time      `json:"approved_at,omitempty"`
	PublishedVersionID    *uuid.UUID      `json:"published_version_id,omitempty"`
	PublishedVersion      string          `json:"published_version,omitempty"`
	IndexRequestID        *uuid.UUID      `json:"index_request_id,omitempty"`
	IndexJobID            *uuid.UUID      `json:"index_job_id,omitempty"`
}
