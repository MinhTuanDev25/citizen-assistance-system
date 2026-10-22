package index

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
)

// HTTPWorker calls POST /v1/index. P4A's Python handler is a mock and does
// not write knowledge_chunks.
// Timeout is INDEX_TIMEOUT_MS. It is not capped at 5s.
type HTTPWorker struct {
	BaseURL string
	Token   string
	Timeout time.Duration
	Client  *http.Client
}

func (w HTTPWorker) Index(ctx context.Context, req Request) (Response, error) {
	if req.SchemaVersion == "" {
		req.SchemaVersion = SchemaVersion
	}
	body, err := marshalWorkerRequest(req)
	if err != nil {
		return Response{}, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(w.BaseURL, "/")+"/v1/index", bytes.NewReader(body))
	if err != nil {
		return Response{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+w.Token)
	client := w.Client
	if client == nil {
		client = &http.Client{Timeout: w.Timeout}
	}
	res, err := client.Do(httpReq)
	if err != nil {
		return Response{}, err
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	if err != nil {
		return Response{}, err
	}
	if res.StatusCode != http.StatusOK {
		return Response{}, fmt.Errorf("index worker status %d", res.StatusCode)
	}
	var out Response
	if err := DecodeStrict(bytes.NewReader(raw), &out); err != nil {
		return Response{}, fmt.Errorf("index worker rejected")
	}
	if (out.SchemaVersion != SchemaVersion && out.SchemaVersion != SchemaVersionV2) || (out.Outcome != OutcomeReady && out.Outcome != OutcomeFailed) {
		return Response{}, fmt.Errorf("index worker rejected")
	}
	if out.SchemaVersion == SchemaVersionV2 && out.Outcome == OutcomeReady && (out.GenerationID == uuid.Nil || out.ChunkCount < 1 || out.ChunkCount != out.VectorCount || out.ManifestHash == "") {
		return Response{}, fmt.Errorf("index worker rejected")
	}
	if out.Outcome == OutcomeReady && out.ErrorCode != nil {
		return Response{}, fmt.Errorf("index worker rejected")
	}
	if out.Outcome == OutcomeFailed && (out.ErrorCode == nil || *out.ErrorCode == "") {
		return Response{}, fmt.Errorf("index worker rejected")
	}
	return out, nil
}

// marshalWorkerRequest sends only the fields the matching Python model accepts.
// The in-memory request carries both contracts, and a zero UUID is not empty
// to encoding/json, so marshaling it directly makes strict v1 and v2 models
// reject the body.
func marshalWorkerRequest(req Request) ([]byte, error) {
	page := req.PageRange
	if page != nil && strings.TrimSpace(*page) == "" {
		page = nil
	}
	if req.SchemaVersion == SchemaVersionV2 {
		return json.Marshal(struct {
			SchemaVersion      string    `json:"schema_version"`
			XAID               string    `json:"xa_id"`
			DocumentID         uuid.UUID `json:"document_id"`
			ProcedureID        uuid.UUID `json:"procedure_id"`
			ProcedureVersionID uuid.UUID `json:"procedure_version_id"`
			JobID              uuid.UUID `json:"job_id"`
			ClaimToken         uuid.UUID `json:"claim_token"`
			GenerationID       uuid.UUID `json:"generation_id"`
			Bucket             string    `json:"bucket"`
			ObjectKey          string    `json:"object_key"`
			Checksum           string    `json:"checksum"`
			PageRange          *string   `json:"page_range,omitempty"`
			PipelineVersion    string    `json:"pipeline_version"`
		}{
			SchemaVersion: req.SchemaVersion, XAID: req.XAID, DocumentID: req.DocumentID,
			ProcedureID: req.ProcedureID, ProcedureVersionID: req.ProcedureVersionID,
			JobID: req.JobID, ClaimToken: req.ClaimToken, GenerationID: req.GenerationID,
			Bucket: req.Bucket, ObjectKey: req.ObjectKey, Checksum: req.Checksum,
			PageRange: page, PipelineVersion: req.PipelineVersion,
		})
	}
	return json.Marshal(struct {
		SchemaVersion      string    `json:"schema_version"`
		DocumentID         uuid.UUID `json:"document_id"`
		XAID               string    `json:"xa_id"`
		DomainID           string    `json:"domain_id"`
		Checksum           string    `json:"checksum"`
		ProcedureVersionID uuid.UUID `json:"procedure_version_id"`
		RelationshipType   string    `json:"relationship_type"`
		PageRange          *string   `json:"page_range,omitempty"`
	}{
		SchemaVersion: req.SchemaVersion, DocumentID: req.DocumentID, XAID: req.XAID,
		DomainID: req.DomainID, Checksum: req.Checksum, ProcedureVersionID: req.ProcedureVersionID,
		RelationshipType: req.RelationshipType, PageRange: page,
	})
}

// DecodeStrict accepts one JSON value and requires the next token to be EOF.
// A second object, a trailing token, or an extra } or ] is rejected.
func DecodeStrict(r io.Reader, dest any) error {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dest); err != nil {
		return err
	}
	var extra json.RawMessage
	err := dec.Decode(&extra)
	if errors.Is(err, io.EOF) {
		return nil
	}
	if err == nil {
		return errors.New("trailing json")
	}
	return err
}
