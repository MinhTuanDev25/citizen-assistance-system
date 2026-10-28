package drafting

import (
	"fmt"
	"strings"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
	"github.com/google/uuid"
)

// CoverageResult is the Wave 2 grounding decision. A single matched quotation
// never establishes global completeness.
type CoverageResult struct {
	QuotesVerified    bool
	CoverageComplete  bool
	GroundingVerified bool
	Matched           int
	Invalid           int
	Missing           int
	Stale             int
	Unknown           int
	Duplicates        int
	MaterialCount     int
	CoveredCount      int
	Issues            []decision.DraftIssue
	Evidence          []EvidenceItem
	Proposals         []MaterialProposal
}

// decided (optional): content keys of proposals an Admin kept without a
// source; they count as covered and are reported as a warning instead.
func EvaluateCoverage(definition []byte, evidence []EvidenceItem, excerpts []Excerpt, source SourceSnapshot, decided ...map[string]bool) CoverageResult {
	out := CoverageResult{
		Issues:    make([]decision.DraftIssue, 0),
		Evidence:  append([]EvidenceItem{}, evidence...),
		Proposals: []MaterialProposal{},
	}
	proposals, err := EnumerateMaterialProposals(definition)
	if err != nil {
		out.Issues = append(out.Issues, decision.DraftIssue{
			Code:     "PARSE_DEFINITION",
			Severity: "error",
			Message:  "cannot enumerate material proposals: " + err.Error(),
		})
		quotes := verifyQuotes(out.Evidence, excerpts, source)
		out.Invalid = quotes.invalid
		out.Matched = quotes.matched
		out.Issues = append(out.Issues, quotes.issues...)
		out.QuotesVerified = quotes.invalid == 0
		out.CoverageComplete = false
		out.GroundingVerified = false
		out.Evidence = quotes.items
		return out
	}
	out.Proposals = proposals
	out.MaterialCount = len(proposals)
	known := proposalsByID(proposals)

	quotes := verifyQuotes(out.Evidence, excerpts, source)
	out.Evidence = quotes.items
	out.Invalid = quotes.invalid
	out.Matched = quotes.matched
	out.Issues = append(out.Issues, quotes.issues...)

	seen := map[string]int{}
	covered := map[string]bool{}
	for i := range out.Evidence {
		item := &out.Evidence[i]
		norm := NormalizeProposalID(item.RawProposalID, known)
		if item.RawProposalID == "" {
			item.RawProposalID = item.ProposalID
			norm = NormalizeProposalID(item.RawProposalID, known)
		}
		if norm.Normalized {
			item.ProposalID = norm.Canonical
			out.Issues = append(out.Issues, decision.DraftIssue{
				Code:       "PROPOSAL_ID_NORMALIZED",
				Severity:   "warning",
				Message:    fmt.Sprintf("normalized %q to %q", norm.Original, norm.Canonical),
				ProposalID: norm.Canonical,
			})
		}
		if norm.Ambiguous {
			out.Unknown++
			item.QuoteValid = false
			item.ReviewStatus = ReviewUnknownProposal
			out.Issues = append(out.Issues, decision.DraftIssue{
				Code:       "AMBIGUOUS_PROPOSAL_ID",
				Severity:   "error",
				Message:    "proposal_id cannot be mapped unambiguously",
				ProposalID: item.RawProposalID,
			})
			continue
		}
		prop, ok := known[item.ProposalID]
		if !ok || norm.Unknown {
			out.Unknown++
			item.QuoteValid = false
			item.ReviewStatus = ReviewUnknownProposal
			out.Issues = append(out.Issues, decision.DraftIssue{
				Code:       "UNKNOWN_PROPOSAL_ID",
				Severity:   "error",
				Message:    "evidence refers to a proposal that is not in the current Definition",
				ProposalID: item.ProposalID,
			})
			continue
		}
		seen[item.ProposalID]++
		if seen[item.ProposalID] > 1 {
			out.Duplicates++
			out.Issues = append(out.Issues, decision.DraftIssue{
				Code:       "DUPLICATE_PROPOSAL_MAPPING",
				Severity:   "error",
				Message:    "duplicate or conflicting evidence mapping for the same proposal",
				ProposalID: item.ProposalID,
			})
			continue
		}
		item.ProposalFingerprint = prop.Fingerprint
		if item.BoundFingerprint != "" && item.BoundFingerprint != prop.Fingerprint {
			out.Stale++
			item.ReviewStatus = ReviewStale
			item.QuoteValid = false
			out.Issues = append(out.Issues, decision.DraftIssue{
				Code:       "STALE_EVIDENCE",
				Severity:   "error",
				Message:    "evidence was bound to a previous revision of this proposal",
				ProposalID: item.ProposalID,
			})
			continue
		}
		if item.BoundFingerprint == "" {
			item.BoundFingerprint = prop.Fingerprint
		}
		if item.QuoteValid {
			covered[item.ProposalID] = true
		}
	}

	adminKept := map[string]bool{}
	for _, d := range decided {
		for k, v := range d {
			adminKept[k] = adminKept[k] || v
		}
	}
	for _, prop := range proposals {
		if !covered[prop.ID] && adminKept[proposalContentKey(prop)] {
			covered[prop.ID] = true
			out.Issues = append(out.Issues, decision.DraftIssue{
				Code:       "ADMIN_DECISION_NO_SOURCE",
				Severity:   "warning",
				Message:    "kept by an Admin decision; the source document does not state it",
				ProposalID: prop.ID,
			})
			continue
		}
		if !covered[prop.ID] {
			out.Missing++
			out.Issues = append(out.Issues, decision.DraftIssue{
				Code:       "MISSING_EVIDENCE",
				Severity:   "error",
				Message:    "material proposal has no valid current source evidence",
				ProposalID: prop.ID,
			})
		}
	}
	out.CoveredCount = len(covered)
	out.QuotesVerified = out.Invalid == 0 && out.Unknown == 0 && out.Duplicates == 0 && out.Stale == 0
	// Vacuous completeness: zero material proposals → coverage complete.
	out.CoverageComplete = out.Missing == 0 && out.Unknown == 0 && out.Duplicates == 0 && out.Stale == 0
	out.GroundingVerified = out.QuotesVerified && out.CoverageComplete
	return out
}

type quoteCheck struct {
	matched int
	invalid int
	issues  []decision.DraftIssue
	items   []EvidenceItem
}

func verifyQuotes(evidence []EvidenceItem, excerpts []Excerpt, source SourceSnapshot) quoteCheck {
	out := quoteCheck{items: append([]EvidenceItem{}, evidence...), issues: make([]decision.DraftIssue, 0)}
	byID := map[uuid.UUID]Excerpt{}
	for _, item := range excerpts {
		byID[item.ExcerptID] = item
	}
	neighbours := newExcerptIndex(excerpts)
	for i := range out.items {
		item := &out.items[i]
		if item.ReviewStatus == "" {
			item.ReviewStatus = ReviewPending
		}
		if item.RawProposalID == "" {
			item.RawProposalID = item.ProposalID
		}
		excerptID, err := uuid.Parse(item.ExcerptID)
		if err != nil {
			out.invalid++
			item.QuoteValid = false
			item.Status = "source_reference_invalid"
			item.ReviewWarning = "invalid excerpt_id"
			out.issues = append(out.issues, decision.DraftIssue{
				Code: "UNKNOWN_EXCERPT_ID", Severity: "error",
				Message: "evidence excerpt_id is not a UUID", ProposalID: item.ProposalID,
			})
			continue
		}
		excerpt, ok := byID[excerptID]
		if !ok {
			out.invalid++
			item.QuoteValid = false
			item.Status = "source_reference_invalid"
			item.ReviewWarning = "unknown excerpt_id"
			out.issues = append(out.issues, decision.DraftIssue{
				Code: "UNKNOWN_EXCERPT_ID", Severity: "error",
				Message: "evidence cites an excerpt that was not in the verified source", ProposalID: item.ProposalID,
			})
			continue
		}
		if source.DocumentID != uuid.Nil && excerpt.DocumentID != uuid.Nil && excerpt.DocumentID != source.DocumentID {
			out.invalid++
			item.QuoteValid = false
			item.Status = "source_reference_invalid"
			out.issues = append(out.issues, decision.DraftIssue{
				Code: "SOURCE_DOCUMENT_MISMATCH", Severity: "error",
				Message: "evidence document does not match the verified source", ProposalID: item.ProposalID,
			})
			continue
		}
		if source.GenerationID != uuid.Nil && excerpt.GenerationID != uuid.Nil && excerpt.GenerationID != source.GenerationID {
			out.invalid++
			item.QuoteValid = false
			item.Status = "source_reference_invalid"
			out.issues = append(out.issues, decision.DraftIssue{
				Code: "SOURCE_GENERATION_MISMATCH", Severity: "error",
				Message: "evidence generation does not match the verified source", ProposalID: item.ProposalID,
			})
			continue
		}
		spansBoundary := !quotationInText(item.Quotation, excerpt.Content) && neighbours.quoteFound(item.Quotation, item.Page, excerpt)
		if !spansBoundary && (item.Page < excerpt.PageStart || item.Page > excerpt.PageEnd) {
			out.invalid++
			item.QuoteValid = false
			item.Status = "source_reference_invalid"
			item.ReviewWarning = "invalid page"
			out.issues = append(out.issues, decision.DraftIssue{
				Code: "INVALID_PAGE", Severity: "error",
				Message: "evidence page is outside the cited excerpt", ProposalID: item.ProposalID,
			})
			continue
		}
		if source.DocumentSHA256 != "" && item.SourceSHA256 != "" && item.SourceSHA256 != source.DocumentSHA256 {
			out.invalid++
			item.QuoteValid = false
			item.Status = "source_reference_invalid"
			out.issues = append(out.issues, decision.DraftIssue{
				Code: "INVALID_SOURCE_HASH", Severity: "error",
				Message: "evidence source hash does not match the verified document", ProposalID: item.ProposalID,
			})
			continue
		}
		if !spansBoundary && !quotationInText(item.Quotation, excerpt.Content) {
			out.invalid++
			item.QuoteValid = false
			item.Status = "quotation_not_found"
			item.ReviewWarning = "quotation not found in excerpt"
			out.issues = append(out.issues, decision.DraftIssue{
				Code: "QUOTATION_NOT_FOUND", Severity: "error",
				Message: "quotation is not present in the cited excerpt", ProposalID: item.ProposalID,
			})
			continue
		}
		item.QuoteValid = true
		item.Status = "source_matched_semantic_unverified"
		if item.ReviewWarning == "" {
			item.ReviewWarning = "source matched; administrative interpretation is not approved"
		}
		out.matched++
		out.issues = append(out.issues, decision.DraftIssue{
			Code: "SOURCE_MATCHED_SEMANTIC_UNVERIFIED", Severity: "warning",
			Message:    "quotation exists in the source; administrative interpretation is not approved",
			ProposalID: item.ProposalID,
		})
	}
	return out
}

func applyCoverage(v *decision.DraftValidation, cov CoverageResult) {
	v.SourceCoverageComplete = cov.CoverageComplete
	v.SourceQuotesVerified = cov.QuotesVerified
	v.GroundingVerified = cov.GroundingVerified && v.SchemaValid
	v.MaterialProposalCount = cov.MaterialCount
	v.CoveredProposalCount = cov.CoveredCount
	v.StaleEvidenceCount = cov.Stale
	v.Issues = append(v.Issues, cov.Issues...)
	if v.SemanticApproved {
		v.SemanticApproved = false
		v.Issues = append(v.Issues, decision.DraftIssue{
			Code: "SEMANTIC_APPROVAL_RESERVED", Severity: "warning",
			Message: "semantic_approved is reserved for explicit Admin approval",
		})
	}
	if !v.SchemaValid {
		v.Readiness = "invalid"
		return
	}
	if v.GroundingVerified {
		v.Readiness = "schema_valid_unapproved"
		return
	}
	v.Readiness = "needs_review"
}

func blockingErrors(issues []decision.DraftIssue) bool {
	for _, issue := range issues {
		if strings.EqualFold(issue.Severity, "error") {
			return true
		}
	}
	return false
}
