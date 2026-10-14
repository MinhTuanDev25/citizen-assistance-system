package activation

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
	"github.com/google/uuid"
)

func evaluateEligibility(row versionRow, sources []sourceRow) Eligibility {
	reasons := make([]string, 0)
	if row.Status == "ACTIVE" {
		return Eligibility{CanActivate: false, Reasons: []string{"VERSION_ALREADY_ACTIVE"}}
	}
	if row.Status == "ARCHIVED" {
		return Eligibility{CanActivate: false, Reasons: []string{"VERSION_ARCHIVED"}}
	}
	if row.Status != "APPROVED" && row.Status != "INDEXING" {
		return Eligibility{CanActivate: false, Reasons: []string{"VERSION_NOT_APPROVED"}}
	}
	if _, err := decision.ParseDefinition(row.Definition); err != nil {
		reasons = append(reasons, "INVALID_DEFINITION")
	} else {
		reasons = append(reasons, definitionIdentityReasons(row)...)
	}
	var sourcesReady int
	var sourceCount int
	for _, src := range sources {
		if src.RelationshipType != "SOURCE" || src.Unlinked {
			continue
		}
		sourceCount++
		itemReasons := sourceReasons(row, src)
		if len(itemReasons) == 0 {
			sourcesReady++
			continue
		}
		reasons = append(reasons, itemReasons...)
	}
	if sourceCount == 0 {
		reasons = append(reasons, "NO_SOURCE_DOCUMENT")
	} else if sourcesReady != sourceCount {
		reasons = append(reasons, "PARTIAL_SOURCE_INDEX")
	}
	reasons = uniqueStrings(reasons)
	return Eligibility{CanActivate: len(reasons) == 0, Reasons: reasons}
}

func sourceReasons(row versionRow, src sourceRow) []string {
	out := make([]string, 0)
	if src.ProcedureID != row.ProcedureID || src.ProcedureVersionID != row.ID {
		out = append(out, "SOURCE_SCOPE_MISMATCH")
	}
	if src.XaID != row.XaID || src.DomainID != row.DomainID {
		out = append(out, "SOURCE_SCOPE_MISMATCH")
	}
	if src.ValidityStatus == "EXPIRED" || src.ValidityStatus == "SUPERSEDED" {
		out = append(out, "DOCUMENT_NOT_CURRENT")
	}
	if src.IndexStatus != "READY" {
		switch src.IndexStatus {
		case "UPLOADED":
			out = append(out, "DOCUMENT_INDEX_UPLOADED")
		case "PROCESSING":
			out = append(out, "DOCUMENT_INDEX_PROCESSING")
		case "FAILED":
			out = append(out, "DOCUMENT_INDEX_FAILED")
		default:
			out = append(out, "DOCUMENT_INDEX_NOT_READY")
		}
		return uniqueStrings(out)
	}
	if src.ActiveGenerationID == nil || *src.ActiveGenerationID == uuid.Nil {
		out = append(out, "ACTIVE_GENERATION_MISSING")
		return uniqueStrings(out)
	}
	if src.GenerationStatus != "READY" {
		out = append(out, "ACTIVE_GENERATION_MISSING")
		return uniqueStrings(out)
	}
	if src.PublishedAt == nil || src.ChunkCount < 1 || src.VectorCount != src.ChunkCount || src.VectorDimension != RequiredVectorDim || strings.TrimSpace(src.EmbeddingModelID) == "" {
		out = append(out, "MOCK_INDEX_NOT_PUBLISHED")
	}
	return uniqueStrings(out)
}

func uniqueStrings(in []string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(in))
	for _, s := range in {
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}

func firstReason(el Eligibility) string {
	if len(el.Reasons) == 0 {
		return "NOT_ELIGIBLE"
	}
	return el.Reasons[0]
}

func documentView(src sourceRow, row versionRow) DocumentLink {
	item := DocumentLink{
		DocumentID:       src.DocumentID,
		Title:            src.Title,
		RelationshipType: src.RelationshipType,
		IndexStatus:      src.IndexStatus,
		ValidityStatus:   src.ValidityStatus,
		ChunkCount:       src.ChunkCount,
		VectorCount:      src.VectorCount,
		VectorDimension:  src.VectorDimension,
		EmbeddingModelID: src.EmbeddingModelID,
		PublishedAt:      src.PublishedAt,
		GenerationStatus: src.GenerationStatus,
	}
	if src.ActiveGenerationID != nil {
		item.ActiveGenerationID = src.ActiveGenerationID
	}
	if src.RelationshipType == "SOURCE" && !src.Unlinked {
		item.Reasons = sourceReasons(row, src)
	}
	return item
}

func definitionIdentityReasons(row versionRow) []string {
	var obj map[string]any
	if err := json.Unmarshal(row.Definition, &obj); err != nil || obj == nil {
		return []string{"INVALID_DEFINITION"}
	}
	out := make([]string, 0)
	if code, ok := jsonString(obj, "procedure_code"); !ok || code != row.ProcedureCode {
		out = append(out, "DEFINITION_IDENTITY_MISMATCH")
	}
	if name, ok := jsonString(obj, "name"); !ok || normalizeName(name) != normalizeName(row.ProcedureName) {
		out = append(out, "DEFINITION_IDENTITY_MISMATCH")
	}
	if domain, ok := jsonString(obj, "domain"); ok && domain != row.DomainID {
		out = append(out, "DEFINITION_IDENTITY_MISMATCH")
	}
	if xaID, ok := jsonString(obj, "xa_id"); ok && xaID != row.XaID {
		out = append(out, "DEFINITION_IDENTITY_MISMATCH")
	}
	if version, ok := jsonString(obj, "version"); ok && version != row.Version {
		out = append(out, "DEFINITION_IDENTITY_MISMATCH")
	}
	return uniqueStrings(out)
}

func jsonString(obj map[string]any, key string) (string, bool) {
	v, ok := obj[key]
	if !ok || v == nil {
		return "", false
	}
	s, ok := v.(string)
	if !ok {
		return "", false
	}
	s = strings.TrimSpace(s)
	if s == "" {
		return "", false
	}
	return s, true
}

func normalizeName(s string) string {
	return strings.Join(strings.Fields(strings.TrimSpace(s)), " ")
}

func wrapDefinition(raw []byte, code, name, domain, xaID, version, status string) ([]byte, error) {
	var obj map[string]any
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, fmt.Errorf("%w: %s", ErrInvalidDefinition, err.Error())
	}
	if obj == nil {
		obj = map[string]any{}
	}
	obj["procedure_code"] = code
	obj["name"] = name
	obj["domain"] = domain
	obj["xa_id"] = xaID
	obj["version"] = version
	obj["status"] = status
	out, err := json.Marshal(obj)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", ErrInvalidDefinition, err.Error())
	}
	if _, err := decision.ParseDefinition(out); err != nil {
		return nil, fmt.Errorf("%w: %s", ErrInvalidDefinition, err.Error())
	}
	return out, nil
}
