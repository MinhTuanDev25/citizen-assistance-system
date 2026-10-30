package chat

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/retrieve"
	"github.com/google/uuid"
)

// Evaluation harness (deploy/runpod/eval), skipped in normal test runs. It feeds
// supplied hits through the production display path, attachHTTPSources: allowed
// source and version filter, relevance floor, page range, lexical overlap,
// classifySupport and selectDisplayCitations. Only the retriever is a stub.
//
//	CAS_DISPLAY_EVAL_IN=in.json CAS_DISPLAY_EVAL_OUT=out.json \
//	  go test ./internal/chat -run TestDisplayEvalHarness -count=1

type displayEvalInput struct {
	XaID          string `json:"xa_id"`
	ProcedureID   string `json:"procedure_id"`
	VersionID     string `json:"procedure_version_id"`
	ProcedureCode string `json:"procedure_code"`
	Pins          []struct {
		DocumentID   string `json:"document_id"`
		GenerationID string `json:"generation_id"`
	} `json:"pins"`
	Cases []struct {
		ID         string         `json:"id"`
		Query      string         `json:"query"`
		Statements []string       `json:"statements"`
		Status     string         `json:"status"`
		Reason     string         `json:"reason"`
		Hits       []retrieve.Hit `json:"hits"`
	} `json:"cases"`
}

type displayEvalCitation struct {
	ChunkID        string   `json:"chunk_id"`
	ContentSHA256  string   `json:"content_sha256"`
	EvidenceStatus string   `json:"evidence_status"`
	PageStart      int      `json:"page_start"`
	RetrievalScore *float64 `json:"retrieval_score"`
}

type displayEvalResult struct {
	ID              string                `json:"id"`
	RetrievalStatus string                `json:"retrieval_status"`
	RetrievalReason string                `json:"retrieval_reason"`
	RequestedTopK   int                   `json:"requested_top_k"`
	Citations       []displayEvalCitation `json:"citations"`
}

type displayEvalRetriever struct {
	resp *retrieve.Response
	got  retrieve.Request
}

func (s *displayEvalRetriever) Retrieve(_ context.Context, req retrieve.Request) (*retrieve.Response, error) {
	s.got = req
	return s.resp, nil
}

func TestDisplayEvalHarness(t *testing.T) {
	inPath, outPath := os.Getenv("CAS_DISPLAY_EVAL_IN"), os.Getenv("CAS_DISPLAY_EVAL_OUT")
	if inPath == "" || outPath == "" {
		t.Skip("evaluation harness: set CAS_DISPLAY_EVAL_IN and CAS_DISPLAY_EVAL_OUT")
	}
	raw, err := os.ReadFile(inPath)
	if err != nil {
		t.Fatal(err)
	}
	var in displayEvalInput
	if err := json.Unmarshal(raw, &in); err != nil {
		t.Fatal(err)
	}
	versionID, procedureID := uuid.MustParse(in.VersionID), uuid.MustParse(in.ProcedureID)
	pins := make([]repository.SourcePin, 0, len(in.Pins))
	for _, p := range in.Pins {
		pins = append(pins, repository.SourcePin{
			DocumentID:         uuid.MustParse(p.DocumentID),
			GenerationID:       uuid.MustParse(p.GenerationID),
			ProcedureID:        procedureID,
			ProcedureVersionID: versionID,
		})
	}
	out := make([]displayEvalResult, 0, len(in.Cases))
	for _, c := range in.Cases {
		plan := &planned{}
		plan.save.VersionID = &versionID
		plan.save.ProcedureID = &procedureID
		plan.save.RequestID = uuid.New()
		plan.result.ProcedureCode = in.ProcedureCode
		stub := &displayEvalRetriever{resp: &retrieve.Response{SchemaVersion: "retrieve.v1", Status: c.Status, Reason: c.Reason, Hits: c.Hits}}
		attachHTTPSources(context.Background(), stub, uuid.New(), in.XaID, c.Query, c.Statements, pins, plan)
		res := displayEvalResult{ID: c.ID, RetrievalStatus: plan.result.RetrievalStatus, RetrievalReason: plan.result.RetrievalReason, RequestedTopK: stub.got.TopK}
		for _, cite := range plan.result.Citations {
			res.Citations = append(res.Citations, displayEvalCitation{
				ChunkID:        cite.ChunkID,
				ContentSHA256:  cite.ContentSHA256,
				EvidenceStatus: cite.EvidenceStatus,
				PageStart:      cite.PageStart,
				RetrievalScore: cite.RetrievalScore,
			})
		}
		out = append(out, res)
	}
	data, err := json.MarshalIndent(out, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(outPath, data, 0o644); err != nil {
		t.Fatal(err)
	}
}
