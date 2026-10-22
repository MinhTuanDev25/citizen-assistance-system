// Package aiextract is the Go side of the P2 LLM Extract contract
// (POST /v1/extract on apps/ai-service). It never trusts the wire response:
// every field is independently re-validated (see validate.go) against the
// same candidates/allowed-keys the Go side itself sent, fail-closed on any
// violation.
package aiextract

import "context"

// SchemaVersion is the only accepted contract version on both sides.
const SchemaVersion = "extract.v1"

// SlotSpec mirrors app/models/extract.py:SlotSpec.
type SlotSpec struct {
	Type       string   `json:"type"`
	Question   string   `json:"question,omitempty"`
	EnumValues []string `json:"enum_values,omitempty"`
}

// Candidate mirrors app/models/extract.py:Candidate.
type Candidate struct {
	ProcedureCode  string              `json:"procedure_code"`
	Name           string              `json:"name"`
	IntentExamples []string            `json:"intent_examples,omitempty"`
	Slots          map[string]SlotSpec `json:"slots,omitempty"`
}

// SlotStateEntry mirrors app/models/extract.py:SlotStateEntry.
type SlotStateEntry struct {
	Value  any    `json:"value"`
	Status string `json:"status"`
}

// PinnedContext mirrors app/models/extract.py:PinnedContext.
type PinnedContext struct {
	ProcedureCode   string                    `json:"procedure_code"`
	AllowedSlotKeys []string                  `json:"allowed_slot_keys"`
	SlotState       map[string]SlotStateEntry `json:"slot_state"`
}

// Request mirrors app/models/extract.py:ExtractRequest.
type Request struct {
	SchemaVersion string         `json:"schema_version"`
	RequestID     string         `json:"request_id"`
	Message       string         `json:"message"`
	Candidates    []Candidate    `json:"candidates"`
	PinnedContext *PinnedContext `json:"pinned_context,omitempty"`
}

// Alternative mirrors app/models/extract.py:Alternative.
type Alternative struct {
	ProcedureCode string  `json:"procedure_code"`
	Confidence    float64 `json:"confidence"`
}

// IntentResult mirrors app/models/extract.py:IntentResult.
type IntentResult struct {
	ProcedureCode *string       `json:"procedure_code"`
	Confidence    float64       `json:"confidence"`
	Alternatives  []Alternative `json:"alternatives,omitempty"`
}

// SlotResult mirrors app/models/extract.py:SlotResult.
type SlotResult struct {
	Key        string  `json:"key"`
	Value      any     `json:"value"`
	Confidence float64 `json:"confidence"`
	Evidence   string  `json:"evidence,omitempty"`
	Operation  string  `json:"operation"`
}

// Response mirrors app/models/extract.py:ExtractResponse. Every field is
// re-validated by Validate() before any of it is trusted.
type Response struct {
	SchemaVersion         string        `json:"schema_version"`
	Intent                *IntentResult `json:"intent"`
	SlotsForProcedureCode *string       `json:"slots_for_procedure_code"`
	Slots                 []SlotResult  `json:"slots"`
	AbstainReason         *string       `json:"abstain_reason"`
	Provider              string        `json:"provider"`
	Model                 string        `json:"model"`
}

// Extractor is implemented by Client (real HTTP) and by test fakes.
type Extractor interface {
	Extract(ctx context.Context, req Request) (*Response, error)
}
