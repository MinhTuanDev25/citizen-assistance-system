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
	// Multi-batch drafting (job). Absent on single-call drafts.
	JobID         *uuid.UUID           `json:"job_id,omitempty"`
	ParentDraftID *uuid.UUID           `json:"parent_draft_id,omitempty"`
	ChunkRanges   []ChunkRange         `json:"source_chunk_ranges,omitempty"`
	Coverage      *DraftCoverage       `json:"coverage,omitempty"`
	Conflicts     []DraftConflict      `json:"conflicts,omitempty"`
	Provenance    []ProposalProvenance `json:"provenance,omitempty"`
	// AdminDecisions: proposals kept without a source quotation on an Admin's
	// stated reason (the document does not say it, e.g. a local practice).
	AdminDecisions []AdminDecision `json:"admin_decisions,omitempty"`
}

type storedPayload struct {
	SchemaVersion         string               `json:"schema_version"`
	Definition            json.RawMessage      `json:"definition"`
	Evidence              []EvidenceItem       `json:"evidence"`
	Warnings              []string             `json:"warnings"`
	MissingRequirements   []string             `json:"missing_requirements"`
	Model                 string               `json:"model"`
	Provider              string               `json:"provider"`
	PromptVersion         string               `json:"prompt_version"`
	PromptHash            string               `json:"prompt_hash"`
	GenerationID          uuid.UUID            `json:"generation_id"`
	DocumentID            uuid.UUID            `json:"document_id"`
	SourcePageStart       int                  `json:"source_page_start"`
	SourcePageEnd         int                  `json:"source_page_end"`
	Usage                 map[string]any       `json:"usage,omitempty"`
	LatencyMS             int                  `json:"latency_ms,omitempty"`
	HistoricalNote        string               `json:"historical_generation_note,omitempty"`
	OriginalProcedureCode string               `json:"original_procedure_code,omitempty"`
	IdentityMismatch      bool                 `json:"identity_mismatch,omitempty"`
	Revision              int                  `json:"revision"`
	ApprovedBy            *uuid.UUID           `json:"approved_by,omitempty"`
	ApprovedAt            *time.Time           `json:"approved_at,omitempty"`
	PublishedVersionID    *uuid.UUID           `json:"published_version_id,omitempty"`
	PublishedVersion      string               `json:"published_version,omitempty"`
	IndexRequestID        *uuid.UUID           `json:"index_request_id,omitempty"`
	IndexJobID            *uuid.UUID           `json:"index_job_id,omitempty"`
	JobID                 *uuid.UUID           `json:"job_id,omitempty"`
	ParentDraftID         *uuid.UUID           `json:"parent_draft_id,omitempty"`
	ChunkRanges           []ChunkRange         `json:"source_chunk_ranges,omitempty"`
	Coverage              *DraftCoverage       `json:"coverage,omitempty"`
	Conflicts             []DraftConflict      `json:"conflicts,omitempty"`
	Provenance            []ProposalProvenance `json:"provenance,omitempty"`
	AdminDecisions        []AdminDecision      `json:"admin_decisions,omitempty"`
}

// AdminDecision keeps one proposal without source evidence. It is pinned to
// the proposal's content (type and text), so editing that content, or the
// proposal moving to another index, is handled: an edited proposal needs a new
// decision, a moved one keeps it.
type AdminDecision struct {
	ProposalID string    `json:"proposal_id"`
	Content    string    `json:"content"`
	Reason     string    `json:"reason"`
	By         uuid.UUID `json:"by"`
	At         time.Time `json:"at"`
}

type AdminDecisionRequest struct {
	ExpectedUpdatedAt *time.Time `json:"expected_updated_at"`
	ProposalID        string     `json:"proposal_id"`
	Reason            string     `json:"reason"`
	// Remove withdraws the decision; the proposal is then missing a source again.
	Remove bool `json:"remove"`
}

// ProvenanceRef is one place in the source that supports a merged proposal.
type ProvenanceRef struct {
	Contributor string `json:"contributor"`
	ExcerptID   string `json:"excerpt_id"`
	ChunkIndex  int    `json:"chunk_index"`
	Page        int    `json:"page"`
	Quotation   string `json:"quotation"`
	QuoteValid  bool   `json:"quote_valid"`
}

// ProposalProvenance lists every batch quotation behind one merged proposal.
// Content pins it to the proposal text it was computed for.
type ProposalProvenance struct {
	ProposalID string          `json:"proposal_id"`
	Content    string          `json:"content"`
	Sources    []ProvenanceRef `json:"sources"`
}

const (
	ConflictOpen     = "OPEN"
	ConflictResolved = "RESOLVED"
)

type ConflictCandidate struct {
	Value       string        `json:"value"`
	Contributor string        `json:"contributor"`
	PageStart   int           `json:"page_start,omitempty"`
	PageEnd     int           `json:"page_end,omitempty"`
	Evidence    *EvidenceItem `json:"evidence,omitempty"`
	// Slot is the full slot definition for a slot-type conflict.
	Slot json.RawMessage `json:"slot,omitempty"`
	// Citation is the full citation for a citations conflict (one document,
	// described differently by different batches).
	Citation json.RawMessage `json:"citation,omitempty"`
	// AlsoFrom: other contributors that proposed exactly this value.
	AlsoFrom []string `json:"also_from,omitempty"`
	// Sources: every quotation behind this candidate; becomes the provenance of
	// the proposal when the Admin chooses it.
	Sources []ProvenanceRef `json:"sources,omitempty"`
}

// DraftConflict: batches stated different values for one legal field. The
// aggregate leaves the field empty; only an Admin decision fills it.
type DraftConflict struct {
	ID         string              `json:"id"`
	Field      string              `json:"field"`
	Status     string              `json:"status"`
	Candidates []ConflictCandidate `json:"candidates"`
	// Chosen is the candidate index, or -1 when the Admin kept the current definition value.
	Chosen     *int       `json:"chosen,omitempty"`
	ResolvedBy *uuid.UUID `json:"resolved_by,omitempty"`
	ResolvedAt *time.Time `json:"resolved_at,omitempty"`
	// ResolvedBasis pins the decision to the field value it produced; when the
	// Definition later says something else the conflict reopens.
	ResolvedBasis string `json:"resolved_basis,omitempty"`
	// Reopened: the value was edited after the decision; the Admin must decide again.
	Reopened bool `json:"reopened,omitempty"`
	// Inactive: a citations conflict of a document the Definition does not cite
	// (merge-dropped or removed). It blocks nothing until the document is cited again.
	Inactive bool `json:"inactive,omitempty"`
}

// CoverageGap: the source has a standard section (e.g. "Lệ phí") but the
// Definition has nothing for the matching field.
type CoverageGap struct {
	Topic        string `json:"topic"`
	Field        string `json:"field"`
	Page         int    `json:"page"`
	Acknowledged bool   `json:"acknowledged,omitempty"`
	Reason       string `json:"reason,omitempty"`
	// Basis identifies this gap for an acknowledgement (topic and source page).
	Basis string `json:"basis"`
}

// GapAck records an Admin decision on a coverage finding and why. Field names
// the finding (a gap field, "truncated:<field>" or "outside_mentions"); Basis
// pins the exact state decided on. When the finding changes or disappears the
// decision is dropped and must be taken again.
type GapAck struct {
	Field  string    `json:"field"`
	Basis  string    `json:"basis"`
	Reason string    `json:"reason"`
	By     uuid.UUID `json:"by"`
	At     time.Time `json:"at"`
}

// Truncation lists merged items dropped because a Definition list hit its limit.
type Truncation struct {
	Field    string   `json:"field"`
	Limit    int      `json:"limit"`
	Dropped  []string `json:"dropped"`
	Blocking bool     `json:"blocking"`
	// DroppedCitations keeps the full dropped citation objects (field
	// "citations"), so an Admin can put one back into the Definition.
	DroppedCitations []json.RawMessage `json:"dropped_citations,omitempty"`
}

type KeepCitationRequest struct {
	ExpectedUpdatedAt *time.Time `json:"expected_updated_at"`
	// Dropped is the index in the citations truncation list.
	Dropped int `json:"dropped"`
	// Replace is the index of a current Definition citation to swap out; nil
	// appends (up to CitationHardMax).
	Replace *int `json:"replace"`
}

// DraftCoverage reports what part of the document a job draft was built from.
type DraftCoverage struct {
	Mode              string     `json:"mode"`
	JobID             uuid.UUID  `json:"job_id"`
	BaseDraftID       *uuid.UUID `json:"base_draft_id,omitempty"`
	Selection         string     `json:"selection"`
	SectionTitle      string     `json:"section_title,omitempty"`
	BoundaryUncertain bool       `json:"boundary_uncertain,omitempty"`
	DocumentPages     int        `json:"document_pages"`
	DocumentChunks    int        `json:"document_chunks"`
	PageStart         int        `json:"page_start"`
	PageEnd           int        `json:"page_end"`
	SelectedChunks    int        `json:"selected_chunks"`
	BaseChunks        int        `json:"base_chunks,omitempty"`
	ProcessedChunks   int        `json:"processed_chunks"`
	Batches           int        `json:"batches"`
	// ScopeComplete: every chunk of the SELECTED scope was drafted. It says
	// nothing about the rest of the document (see OutsideMentionsReviewed).
	ScopeComplete bool `json:"scope_complete"`
	// Complete mirrors ScopeComplete for drafts stored before it existed.
	Complete bool `json:"complete"`
	// OutsideMentionsReviewed: every place OutsideCriteria found outside every
	// procedure section was reviewed by an Admin (or none was found). It is NOT
	// proof that the document holds no other applicable rule: the search only
	// covers OutsideCriteria. Computed on every validation.
	OutsideMentionsReviewed bool `json:"outside_mentions_reviewed"`
	// OutsideCriteria states what the outside search looked for.
	OutsideCriteria string `json:"outside_criteria,omitempty"`
	// BoundaryReviewed: the section boundary was flagged uncertain and an Admin
	// confirmed it (false while that decision is pending).
	BoundaryReviewed bool `json:"boundary_reviewed,omitempty"`
	// Truncated: merged items dropped by the Definition size limits.
	Truncated         []Truncation  `json:"truncated,omitempty"`
	DuplicatesRemoved int           `json:"duplicates_removed"`
	Gaps              []CoverageGap `json:"gaps"`
	Acknowledged      []GapAck      `json:"acknowledged,omitempty"`
	// OutsideMentions: places outside the drafted scope that name the procedure.
	// They are listed for the Admin, never merged automatically.
	OutsideMentions []OutsideMention `json:"outside_mentions,omitempty"`
}

// OutsideMention groups chunks outside the drafted scope that contain the
// procedure name, by the section they belong to ("" = no procedure section).
type OutsideMention struct {
	// Kind: "name" (chunk names the procedure) or "general" (a chunk outside
	// every section with general-rule keywords: fees, deadlines, forms...).
	Kind      string `json:"kind"`
	Section   string `json:"section"`
	PageStart int    `json:"page_start"`
	PageEnd   int    `json:"page_end"`
	Chunks    int    `json:"chunks"`
	// Pages: the exact pages, as contiguous spans (e.g. 2, 100, 450 stay apart).
	Pages []PageSpan `json:"pages"`
	// ChunkIndexes identify the exact chunks; the review decision is pinned to them.
	ChunkIndexes []int `json:"chunk_indexes"`
}

type PageSpan struct {
	Start int `json:"start"`
	End   int `json:"end"`
}

type AcknowledgeGapRequest struct {
	ExpectedUpdatedAt *time.Time `json:"expected_updated_at"`
	Field             string     `json:"field"`
	Reason            string     `json:"reason"`
}

type ResolveConflictRequest struct {
	ExpectedUpdatedAt *time.Time `json:"expected_updated_at"`
	ConflictID        string     `json:"conflict_id"`
	Choice            *int       `json:"choice"`
}
