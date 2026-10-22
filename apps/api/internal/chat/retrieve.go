package chat

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"
	"strings"
	"unicode"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/retrieve"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

const (
	liveSearchK       = 12
	liveDisplayK      = 3
	liveSourceLoadCap = 40
	liveSnippetRunes  = 280
	liveQueryMaxRunes = 2000
)

// attachLiveSources freezes pinned-scope PDF evidence with the procedure
// version, then (on guidance turns only) attaches related passages.
// Checklist stays on the Decision Engine. Retrieval never switches a session
// onto a later version or generation. HTTP retrieve errors leave the
// definition reply unchanged.
func attachLiveSources(ctx context.Context, tx pgx.Tx, sessions *repository.SessionRepo, retriever retrieve.Retriever, sessionID uuid.UUID, xaID string, question string, plan *planned) {
	if plan == nil || plan.save.VersionID == nil || plan.save.ProcedureID == nil || sessions == nil {
		return
	}
	pins, err := sessions.EnsureSourcePins(ctx, tx, sessionID, xaID, *plan.save.ProcedureID, *plan.save.VersionID)
	if err != nil {
		if isGuidanceAction(plan.result.Action) {
			setRetrievalStatus(plan, "retrieval_unavailable")
		}
		return
	}
	if !isGuidanceAction(plan.result.Action) {
		return
	}
	query := retrievalQuery(question, plan)
	if plan.save.Metadata == nil {
		plan.save.Metadata = map[string]any{}
	}
	plan.save.Metadata["retrieval_query"] = query
	statements := checklistStatements(plan)
	if retriever != nil {
		attachHTTPSources(ctx, retriever, sessionID, xaID, query, statements, pins, plan)
		return
	}
	if len(pins) == 0 {
		setRetrievalStatus(plan, "no_relevant_evidence")
		return
	}
	rows, err := sessions.LoadPinnedChunks(ctx, tx, xaID, *plan.save.VersionID, pins, liveSourceLoadCap)
	if err != nil || len(rows) == 0 {
		if err != nil {
			setRetrievalStatus(plan, "retrieval_unavailable")
			return
		}
		setRetrievalStatus(plan, "no_relevant_evidence")
		return
	}
	hits := pickPinnedPassagesByTerms(relatednessTerms(query, statements), rows, liveSearchK)
	if len(hits) == 0 {
		setRetrievalStatus(plan, "no_relevant_evidence")
		return
	}
	cites, saved := hitsToCitations(hits, *plan.save.VersionID, statements, plan.result.ProcedureCode)
	cites, saved = selectDisplayCitations(cites, saved, liveDisplayK)
	applyEvidence(plan, cites, saved, "ok", "")
}

func isGuidanceAction(action string) bool {
	return action == "DIRECT_ANSWER" || action == "PROVIDE_FINAL_GUIDANCE"
}

func attachHTTPSources(ctx context.Context, retriever retrieve.Retriever, sessionID uuid.UUID, xaID, question string, statements []string, pins []repository.SourcePin, plan *planned) {
	sources := make([]retrieve.Source, 0, len(pins))
	for _, pin := range pins {
		sources = append(sources, retrieve.Source{
			DocumentID:   pin.DocumentID.String(),
			GenerationID: pin.GenerationID.String(),
		})
	}
	truncated := false
	if len(sources) > retrieve.MaxAllowedSources {
		slog.Warn("retrieve allowed_sources capped",
			"request_id", plan.save.RequestID.String(),
			"session_id", sessionID.String(),
			"pinned", len(sources),
			"sent", retrieve.MaxAllowedSources,
		)
		sources = sources[:retrieve.MaxAllowedSources]
		truncated = true
	}
	req := retrieve.Request{
		Query:              question,
		XaID:               xaID,
		ProcedureID:        plan.save.ProcedureID.String(),
		ProcedureVersionID: plan.save.VersionID.String(),
		AllowedSources:     sources,
		TopK:               liveSearchK,
		Statements:         statements,
		RequestID:          plan.save.RequestID.String(),
	}
	resp, err := retriever.Retrieve(ctx, req)
	if err != nil {
		reason := retrieve.UnavailableReason(err)
		slog.Warn("retrieve unavailable",
			"request_id", req.RequestID,
			"session_id", sessionID.String(),
			"xa_id", xaID,
			"reason", reason,
			"err", err,
		)
		setRetrievalOutcome(plan, "retrieval_unavailable", reason)
		return
	}
	status := mapRetrieveStatus(resp.Status, resp.Reason)
	reason := resp.Reason
	if truncated && status != "retrieval_unavailable" && status != "retrieval_incompatible" {
		reason = "allowed_sources_truncated"
	}
	if status != "ok" || len(resp.Hits) == 0 {
		if status == "ok" {
			status = "no_relevant_evidence"
		}
		setRetrievalOutcome(plan, status, reason)
		return
	}
	allowed := map[string]struct{}{}
	for _, pin := range pins {
		allowed[pin.DocumentID.String()+"/"+pin.GenerationID.String()] = struct{}{}
	}
	cites := make([]decision.Citation, 0, len(resp.Hits))
	saved := make([]map[string]any, 0, len(resp.Hits))
	for _, hit := range resp.Hits {
		if _, ok := allowed[hit.DocumentID+"/"+hit.GenerationID]; !ok {
			continue
		}
		if hit.ProcedureVersionID != "" && hit.ProcedureVersionID != plan.save.VersionID.String() {
			continue
		}
		if !retrieve.PassesRelatedFloor(hit.RetrievalScore) {
			continue
		}
		if hit.PageStart < 1 || hit.PageEnd < hit.PageStart {
			continue
		}
		if overlap(relatednessTerms(question, statements), hit.Excerpt) <= 0 {
			continue
		}
		excerpt := snippet(hit.Excerpt, liveSnippetRunes)
		sum := sha256.Sum256([]byte(excerpt))
		title, number := pinSnapshotTitle(hit.Title, hit.DocumentNumber, hit.DocumentID, pins)
		page := formatPage(hit.PageStart, hit.PageEnd)
		status := classifySupport(hit, statements, plan.result.ProcedureCode)
		cite := decision.Citation{
			DocID:            hit.DocumentID,
			Title:            title,
			SourceType:       "indexed_pdf",
			PageHint:         page,
			Excerpt:          excerpt,
			EvidenceStatus:   status,
			GenerationID:     hit.GenerationID,
			ChunkID:          hit.ChunkID,
			DocumentID:       hit.DocumentID,
			DocumentNumber:   number,
			Availability:     orDefault(hit.Availability, "available"),
			RetrievalScore:   hit.RetrievalScore,
			PageStart:        hit.PageStart,
			PageEnd:          hit.PageEnd,
			ExcerptSHA256:    hex.EncodeToString(sum[:]),
			ContentSHA256:    hit.ContentSHA256,
			ProcedureVersion: hit.ProcedureVersionID,
		}
		cites = append(cites, cite)
		saved = append(saved, citationSave(cite, sessionID))
	}
	cites, saved = selectDisplayCitations(cites, saved, liveDisplayK)
	if len(cites) == 0 {
		if truncated {
			reason = "allowed_sources_truncated"
		}
		setRetrievalOutcome(plan, "no_relevant_evidence", reason)
		return
	}
	applyEvidence(plan, cites, saved, "ok", reason)
}

func applyEvidence(plan *planned, cites []decision.Citation, saved []map[string]any, status, reason string) {
	setRetrievalOutcome(plan, status, reason)
	if len(cites) == 0 {
		return
	}
	plan.result.Citations = append(append([]decision.Citation{}, plan.result.Citations...), cites...)
	plan.save.Citations = append(append([]map[string]any{}, plan.save.Citations...), saved...)
	if plan.save.Metadata == nil {
		plan.save.Metadata = map[string]any{}
	}
	plan.save.Metadata["citations"] = plan.result.Citations
	plan.save.Metadata["source_passages"] = len(cites)
	plan.save.Metadata["retrieval_status"] = status
}

func selectDisplayCitations(cites []decision.Citation, saved []map[string]any, limit int) ([]decision.Citation, []map[string]any) {
	if limit <= 0 || len(cites) == 0 {
		return nil, nil
	}
	type ranked struct {
		cite       decision.Citation
		saved      map[string]any
		supporting bool
		order      int
	}
	rows := make([]ranked, 0, len(cites))
	for i, cite := range cites {
		row := ranked{cite: cite, supporting: cite.EvidenceStatus == "supporting", order: i}
		if i < len(saved) {
			row.saved = saved[i]
		}
		rows = append(rows, row)
	}
	for i := 0; i < len(rows); i++ {
		best := i
		for j := i + 1; j < len(rows); j++ {
			if rows[j].supporting && !rows[best].supporting {
				best = j
				continue
			}
			if rows[j].supporting == rows[best].supporting && rows[j].order < rows[best].order {
				best = j
			}
		}
		rows[i], rows[best] = rows[best], rows[i]
	}
	outC := make([]decision.Citation, 0, limit)
	outS := make([]map[string]any, 0, limit)
	for _, row := range rows {
		outC = append(outC, row.cite)
		outS = append(outS, row.saved)
		if len(outC) >= limit {
			break
		}
	}
	return outC, outS
}

func hitsToCitations(hits []repository.PinnedChunk, versionID uuid.UUID, statements []string, procedureCode string) ([]decision.Citation, []map[string]any) {
	cites := make([]decision.Citation, 0, len(hits))
	saved := make([]map[string]any, 0, len(hits))
	for _, hit := range hits {
		excerpt := snippet(hit.Content, liveSnippetRunes)
		status := classifySupport(retrieve.Hit{
			DocumentID:    hit.DocumentID.String(),
			GenerationID:  hit.GenerationID.String(),
			PageStart:     hit.PageStart,
			PageEnd:       hit.PageEnd,
			Excerpt:       hit.Content,
			ContentSHA256: hit.TextSHA256,
		}, statements, procedureCode)
		page := formatPage(hit.PageStart, hit.PageEnd)
		sum := sha256.Sum256([]byte(excerpt))
		cite := decision.Citation{
			DocID:            hit.DocumentID.String(),
			Title:            hit.Title,
			SourceType:       "indexed_pdf",
			PageHint:         page,
			Excerpt:          excerpt,
			EvidenceStatus:   status,
			GenerationID:     hit.GenerationID.String(),
			ChunkID:          hit.ChunkID.String(),
			DocumentID:       hit.DocumentID.String(),
			DocumentNumber:   hit.DocumentNumber,
			Availability:     "available",
			PageStart:        hit.PageStart,
			PageEnd:          hit.PageEnd,
			ExcerptSHA256:    hex.EncodeToString(sum[:]),
			ContentSHA256:    hit.TextSHA256,
			ProcedureVersion: versionID.String(),
		}
		cites = append(cites, cite)
		saved = append(saved, citationSave(cite, uuid.Nil))
	}
	return cites, saved
}

func citationSave(cite decision.Citation, _ uuid.UUID) map[string]any {
	row := map[string]any{
		"doc_id":               cite.DocID,
		"title":                cite.Title,
		"source_type":          cite.SourceType,
		"page_range":           cite.PageHint,
		"excerpt":              cite.Excerpt,
		"excerpt_sha256":       cite.ExcerptSHA256,
		"generation_id":        cite.GenerationID,
		"procedure_version_id": cite.ProcedureVersion,
		"evidence_status":      cite.EvidenceStatus,
		"content_sha256":       cite.ContentSHA256,
		"page_start":           cite.PageStart,
		"page_end":             cite.PageEnd,
		"availability":         cite.Availability,
		"document_id":          cite.DocumentID,
		"knowledge_chunk_id":   cite.ChunkID,
	}
	if cite.RetrievalScore != nil {
		row["retrieval_score"] = *cite.RetrievalScore
	}
	return row
}

func setRetrievalStatus(plan *planned, status string) {
	setRetrievalOutcome(plan, status, "")
}

func setRetrievalOutcome(plan *planned, status, reason string) {
	plan.result.RetrievalStatus = status
	plan.result.RetrievalReason = reason
	if plan.save.Metadata == nil {
		plan.save.Metadata = map[string]any{}
	}
	plan.save.Metadata["retrieval_status"] = status
	if reason != "" {
		plan.save.Metadata["retrieval_reason"] = reason
	}
}

func mapRetrieveStatus(status, reason string) string {
	switch status {
	case "ok":
		return "ok"
	case "no_relevant_evidence":
		return "no_relevant_evidence"
	case "all_candidates_rejected":
		return "all_candidates_rejected"
	case "retrieval_incompatible":
		return "retrieval_incompatible"
	case "retrieval_unavailable":
		return "retrieval_unavailable"
	default:
		if reason == "embedding_identity_mismatch" {
			return "retrieval_incompatible"
		}
		return "retrieval_unavailable"
	}
}

func checklistStatements(plan *planned) []string {
	if plan == nil || plan.result.Guidance == nil {
		return nil
	}
	return append([]string{}, plan.result.Guidance.Checklist...)
}

// retrievalQuery is the E5 search string. Guidance turns use the pinned
// Definition (procedure name, summary, checklist), not the last slot-fill
// utterance. The Decision Engine reply is unchanged. Last user text is only
// a fallback when no Definition language exists (empty pins / no checklist).
func retrievalQuery(message string, plan *planned) string {
	parts := make([]string, 0, 8)
	if plan != nil {
		if name := strings.TrimSpace(plan.procedureName); name != "" {
			parts = append(parts, name)
		}
		if code := strings.TrimSpace(plan.result.ProcedureCode); code != "" {
			parts = append(parts, code)
		}
		if g := plan.result.Guidance; g != nil {
			if summary := strings.TrimSpace(g.Summary); summary != "" {
				parts = append(parts, summary)
			}
			for _, item := range g.Checklist {
				if s := strings.TrimSpace(item); s != "" {
					parts = append(parts, s)
				}
			}
		}
	}
	joined := strings.Join(strings.Fields(strings.Join(parts, " ")), " ")
	if joined != "" {
		return clipRunes(joined, liveQueryMaxRunes)
	}
	return clipRunes(strings.Join(strings.Fields(strings.TrimSpace(message)), " "), liveQueryMaxRunes)
}

func clipRunes(s string, max int) string {
	if max <= 0 {
		return s
	}
	runes := []rune(s)
	if len(runes) <= max {
		return s
	}
	return string(runes[:max])
}

func classifySupport(hit retrieve.Hit, statements []string, procedureCode string) string {
	return retrieve.ClassifySupport(hit, statements, procedureCode)
}

func pinSnapshotTitle(title, number, documentID string, pins []repository.SourcePin) (string, string) {
	for _, pin := range pins {
		if pin.DocumentID.String() != documentID {
			continue
		}
		if pin.Title != "" {
			title = pin.Title
		}
		if pin.DocumentNumber != "" {
			number = pin.DocumentNumber
		}
		return title, number
	}
	return title, number
}

func formatPage(start, end int) string {
	if start <= 0 {
		return ""
	}
	if end > start {
		return fmt.Sprintf("%d–%d", start, end)
	}
	return fmt.Sprintf("%d", start)
}

func orDefault(v, fallback string) string {
	if strings.TrimSpace(v) == "" {
		return fallback
	}
	return v
}

func relatednessTerms(question string, statements []string) []string {
	var terms []string
	for _, item := range statements {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		terms = append(terms, tokens(item)...)
	}
	if len(terms) > 0 {
		return terms
	}
	return tokens(question)
}

func pickPinnedPassages(question string, rows []repository.PinnedChunk, limit int) []repository.PinnedChunk {
	return pickPinnedPassagesByTerms(tokens(question), rows, limit)
}

func pickPinnedPassagesByTerms(terms []string, rows []repository.PinnedChunk, limit int) []repository.PinnedChunk {
	if limit <= 0 || len(rows) == 0 {
		return nil
	}
	scored := append([]repository.PinnedChunk{}, rows...)
	for i := range scored {
		scored[i].Score = overlap(terms, scored[i].Content)
	}
	for i := 0; i < len(scored); i++ {
		best := i
		for j := i + 1; j < len(scored); j++ {
			if scored[j].Score > scored[best].Score {
				best = j
			}
		}
		scored[i], scored[best] = scored[best], scored[i]
	}
	out := make([]repository.PinnedChunk, 0, limit)
	for _, row := range scored {
		if row.Score <= 0 {
			continue
		}
		out = append(out, row)
		if len(out) >= limit {
			break
		}
	}
	return out
}

func pickPassages(question string, rows []sourceChunk, limit int) []sourceChunk {
	pinned := make([]repository.PinnedChunk, len(rows))
	for i, row := range rows {
		pinned[i] = repository.PinnedChunk{Title: row.Title, Content: row.Content, PageStart: row.PageStart, PageEnd: row.PageEnd, Score: row.Score}
	}
	got := pickPinnedPassages(question, pinned, limit)
	out := make([]sourceChunk, len(got))
	for i, row := range got {
		out[i] = sourceChunk{Title: row.Title, Content: row.Content, PageStart: row.PageStart, PageEnd: row.PageEnd, Score: row.Score}
	}
	return out
}

type sourceChunk struct {
	DocumentID uuid.UUID
	Title      string
	Content    string
	PageStart  int
	PageEnd    int
	Score      int
}

func tokens(s string) []string {
	fields := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return unicode.IsSpace(r) || unicode.IsPunct(r)
	})
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		if len([]rune(f)) < 2 {
			continue
		}
		out = append(out, f)
	}
	return out
}

func overlap(terms []string, content string) int {
	if len(terms) == 0 {
		return 0
	}
	folded := strings.ToLower(content)
	n := 0
	for _, t := range terms {
		if strings.Contains(folded, t) {
			n++
		}
	}
	return n
}

func snippet(s string, maxRunes int) string {
	cleaned := strings.Join(strings.Fields(s), " ")
	runes := []rune(cleaned)
	if len(runes) <= maxRunes {
		return cleaned
	}
	return string(runes[:maxRunes]) + "…"
}
