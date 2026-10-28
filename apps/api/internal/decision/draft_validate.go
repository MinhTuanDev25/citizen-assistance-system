package decision

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/google/uuid"
)

// Allowed slot types used by the current Decision Engine runtime.
var AllowedSlotTypes = map[string]struct{}{
	"string":  {},
	"boolean": {},
	"number":  {},
	"enum":    {},
}

// ForbiddenDefinitionKeys must not appear on an AI draft marked schema-valid.
var ForbiddenDefinitionKeys = []string{"support_map", "approved_support_map"}

// MaxDraftCitations bounds the citations of an AI-drafted Definition (the draft
// review → approve path). The model emits at most 10; an Admin may keep
// merge-dropped ones up to this limit. It is NOT a system-wide Definition rule:
// versions created or edited directly through the version API are not limited
// (activation only parses the Definition). Citizen replies list every
// citation, so the bound also keeps drafted replies readable.
const MaxDraftCitations = 20

func sortedSlotKeys(slots map[string]SlotDef) []string {
	keys := make([]string, 0, len(slots))
	for key := range slots {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// guidanceConditionProblem: a conditional item must name an existing
// enum/boolean question and one of its answers, and say something.
func guidanceConditionProblem(def Definition, item GuidanceItem) string {
	if strings.TrimSpace(item.Item) == "" {
		return "item is empty"
	}
	slot, ok := def.Slots[item.When.Slot]
	if !ok {
		return fmt.Sprintf("when.slot %q is not a question of this procedure", item.When.Slot)
	}
	var want any
	if err := json.Unmarshal(item.When.Equals, &want); err != nil {
		return "when.equals is missing or invalid"
	}
	switch slot.Type {
	case "enum":
		s, isString := want.(string)
		for _, v := range slot.EnumValues {
			if isString && v == s {
				return ""
			}
		}
		return fmt.Sprintf("when.equals %s is not an answer of %s", string(item.When.Equals), item.When.Slot)
	case "boolean":
		if _, isBool := want.(bool); isBool {
			return ""
		}
		return "when.equals must be true or false for a yes/no question"
	default:
		return fmt.Sprintf("conditions can only use enum or yes/no questions, not %s", slot.Type)
	}
}

// NormalizeSourceID gives one spelling per cited document: a UUID in any form
// uuid.Parse accepts (hyphenated, 32 hex, braces, urn, any case) becomes its
// canonical form; any other ID is trimmed and lower-cased.
func NormalizeSourceID(id string) string {
	id = strings.TrimSpace(id)
	if u, err := uuid.Parse(id); err == nil {
		return u.String()
	}
	return strings.ToLower(id)
}

type DraftIssue struct {
	Code       string `json:"code"`
	Severity   string `json:"severity"`
	Message    string `json:"message"`
	ProposalID string `json:"proposal_id,omitempty"`
}

type DraftValidation struct {
	SchemaValid            bool         `json:"schema_valid"`
	GroundingVerified      bool         `json:"grounding_verified"`
	SourceCoverageComplete bool         `json:"source_coverage_complete"`
	SourceQuotesVerified   bool         `json:"source_quotes_verified"`
	SemanticApproved       bool         `json:"semantic_approved"`
	IdentityMismatch       bool         `json:"identity_mismatch,omitempty"`
	Issues                 []DraftIssue `json:"issues"`
	Readiness              string       `json:"readiness"`
	ParsedProcedure        string       `json:"parsed_procedure_code,omitempty"`
	ChecklistCount         int          `json:"checklist_count"`
	SlotCount              int          `json:"slot_count"`
	RequiredSlotCount      int          `json:"required_slot_count"`
	ConditionalCount       int          `json:"conditional_count"`
	MaterialProposalCount  int          `json:"material_proposal_count"`
	CoveredProposalCount   int          `json:"covered_proposal_count"`
	StaleEvidenceCount     int          `json:"stale_evidence_count"`
	ProceduralReady        bool         `json:"procedural_ready"`
}

// ValidateDraftDefinition runs ParseDefinition plus the extra structural
// checks needed so an AI draft cannot be labeled valid when slot/condition
// references are broken. It does not activate versions and never sets
// SemanticApproved.
func ValidateDraftDefinition(raw []byte, expectedCode string) DraftValidation {
	out := DraftValidation{
		SemanticApproved: false,
		Readiness:        "invalid",
		Issues:           []DraftIssue{},
	}
	for _, key := range extraTopLevelKeys(raw) {
		out.Issues = append(out.Issues, DraftIssue{
			Code:     "FORBIDDEN_DEFINITION_KEY",
			Severity: "error",
			Message:  "definition contains unapproved key " + key,
		})
	}
	def, err := ParseDefinition(raw)
	if err != nil {
		out.Issues = append(out.Issues, DraftIssue{
			Code:     "PARSE_DEFINITION",
			Severity: "error",
			Message:  err.Error(),
		})
		return out
	}
	out.ParsedProcedure = def.ProcedureCode
	out.SlotCount = len(def.Slots)
	out.RequiredSlotCount = len(def.RequiredSlots)
	out.ConditionalCount = len(def.Conditional)
	out.ChecklistCount = len(def.Guidance.Checklist)
	if len(def.Citations) > MaxDraftCitations {
		out.Issues = append(out.Issues, DraftIssue{
			Code:     "TOO_MANY_CITATIONS",
			Severity: "error",
			Message:  fmt.Sprintf("definition has %d citations; at most %d are allowed", len(def.Citations), MaxDraftCitations),
		})
	}
	for i, c := range def.Citations {
		docID, documentID := strings.TrimSpace(c.DocID), strings.TrimSpace(c.DocumentID)
		if docID != "" && documentID != "" && NormalizeSourceID(docID) != NormalizeSourceID(documentID) {
			out.Issues = append(out.Issues, DraftIssue{
				Code:     "CITATION_SOURCE_CONFLICT",
				Severity: "error",
				Message:  fmt.Sprintf("citation %d has doc_id %q and document_id %q; they must name the same document", i+1, docID, documentID),
			})
		}
	}

	// Every question must be reachable: asked always (required_slots) or when
	// a conditional rule fires. Otherwise the citizen is never asked it, and
	// anything that depends on its answer never shows.
	reachable := map[string]bool{}
	for _, key := range def.RequiredSlots {
		reachable[key] = true
	}
	for _, c := range def.Conditional {
		for _, key := range c.Require {
			reachable[key] = true
		}
	}
	for _, key := range sortedSlotKeys(def.Slots) {
		if !reachable[key] {
			out.Issues = append(out.Issues, DraftIssue{
				Code:       "UNREACHABLE_SLOT",
				Severity:   "error",
				Message:    fmt.Sprintf("slot %s is neither required nor required by a conditional rule, so it is never asked", key),
				ProposalID: "slot:" + key,
			})
		}
	}
	for field, items := range map[string][]GuidanceItem{
		"conditional_checklist": def.Guidance.ConditionalChecklist,
		"conditional_notes":     def.Guidance.ConditionalNotes,
	} {
		for i, item := range items {
			if msg := guidanceConditionProblem(def, item); msg != "" {
				out.Issues = append(out.Issues, DraftIssue{
					Code:       "INVALID_GUIDANCE_CONDITION",
					Severity:   "error",
					Message:    fmt.Sprintf("guidance.%s[%d]: %s", field, i, msg),
					ProposalID: fmt.Sprintf("%s:%d", field, i),
				})
			}
		}
	}

	if expectedCode != "" && def.ProcedureCode != expectedCode {
		out.Issues = append(out.Issues, DraftIssue{
			Code:     "PROCEDURE_CODE_MISMATCH",
			Severity: "error",
			Message:  fmt.Sprintf("procedure_code %q does not match %q", def.ProcedureCode, expectedCode),
		})
	}
	if strings.TrimSpace(def.Name) == "" {
		out.Issues = append(out.Issues, DraftIssue{
			Code:     "MISSING_NAME",
			Severity: "error",
			Message:  "name is required",
		})
	}
	for key, slot := range def.Slots {
		if _, ok := AllowedSlotTypes[slot.Type]; !ok {
			out.Issues = append(out.Issues, DraftIssue{
				Code:       "INVALID_SLOT_TYPE",
				Severity:   "error",
				Message:    fmt.Sprintf("slot %s has unsupported type %q", key, slot.Type),
				ProposalID: "slot:" + key,
			})
		}
		if len(slot.EnumLabels) > 0 && (slot.Type != "enum" || len(slot.EnumLabels) != len(slot.EnumValues)) {
			out.Issues = append(out.Issues, DraftIssue{
				Code:       "ENUM_LABELS_MISMATCH",
				Severity:   "error",
				Message:    fmt.Sprintf("slot %s has %d enum_labels for %d enum_values", key, len(slot.EnumLabels), len(slot.EnumValues)),
				ProposalID: "slot:" + key,
			})
		}
		if slot.Type == "enum" && len(slot.EnumValues) == 0 {
			out.Issues = append(out.Issues, DraftIssue{
				Code:       "ENUM_VALUES_REQUIRED",
				Severity:   "error",
				Message:    "enum slot requires enum_values",
				ProposalID: "slot:" + key,
			})
		}
		if strings.TrimSpace(slot.Question) == "" {
			out.Issues = append(out.Issues, DraftIssue{
				Code:       "MISSING_SLOT_QUESTION",
				Severity:   "error",
				Message:    "slot question is required",
				ProposalID: "slot:" + key,
			})
		}
	}
	for _, key := range def.RequiredSlots {
		if _, ok := def.Slots[key]; !ok {
			out.Issues = append(out.Issues, DraftIssue{
				Code:       "UNKNOWN_REQUIRED_SLOT",
				Severity:   "error",
				Message:    "required_slots references unknown slot " + key,
				ProposalID: "required:" + key,
			})
		}
	}
	for i, rule := range def.Conditional {
		if _, ok := def.Slots[rule.WhenSlot]; !ok {
			out.Issues = append(out.Issues, DraftIssue{
				Code:       "UNKNOWN_CONDITION_SLOT",
				Severity:   "error",
				Message:    "conditional when.slot is unknown",
				ProposalID: fmt.Sprintf("conditional:%d", i),
			})
		}
		for _, key := range rule.Require {
			if _, ok := def.Slots[key]; !ok {
				out.Issues = append(out.Issues, DraftIssue{
					Code:       "UNKNOWN_CONDITION_REQUIRE",
					Severity:   "error",
					Message:    "conditional require references unknown slot " + key,
					ProposalID: fmt.Sprintf("conditional:%d", i),
				})
			}
		}
	}
	out.SchemaValid = !hasError(out.Issues)
	readyIssues := proceduralReadiness(def)
	out.Issues = append(out.Issues, readyIssues...)
	out.ProceduralReady = out.SchemaValid && !hasError(readyIssues)
	if out.SchemaValid {
		out.Readiness = "schema_valid_unapproved"
	}
	return out
}

// proceduralReadiness is independent of source-coverage. An empty Definition
// may be vacuously covered and still unusable for citizen guidance.
func proceduralReadiness(def Definition) []DraftIssue {
	var issues []DraftIssue
	usableSlots := 0
	for key, slot := range def.Slots {
		if strings.TrimSpace(slot.Question) != "" {
			usableSlots++
			continue
		}
		issues = append(issues, DraftIssue{
			Code:       "SLOT_QUESTION_REQUIRED",
			Severity:   "error",
			Message:    "slot is missing a citizen-facing question",
			ProposalID: "slot:" + key,
		})
	}
	if usableSlots == 0 {
		issues = append(issues, DraftIssue{
			Code:     "NO_CITIZEN_SLOT",
			Severity: "error",
			Message:  "definition has no usable slot for citizen guidance",
		})
	}
	hasGuidance := strings.TrimSpace(def.Guidance.Summary) != "" ||
		len(def.Guidance.Checklist) > 0 ||
		strings.TrimSpace(def.Guidance.WhereToSubmit) != ""
	if !hasGuidance {
		issues = append(issues, DraftIssue{
			Code:     "NO_GUIDANCE_PATH",
			Severity: "error",
			Message:  "definition needs summary, checklist, or where_to_submit",
		})
	}
	return issues
}

func extraTopLevelKeys(raw []byte) []string {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil
	}
	var found []string
	for _, key := range ForbiddenDefinitionKeys {
		if _, ok := m[key]; ok {
			found = append(found, key)
		}
	}
	return found
}

func hasError(issues []DraftIssue) bool {
	for _, issue := range issues {
		if issue.Severity == "error" {
			return true
		}
	}
	return false
}
