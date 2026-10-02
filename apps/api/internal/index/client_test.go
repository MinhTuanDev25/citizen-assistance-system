package index

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
)

func TestHTTPWorkerRejectsChunkPayload(t *testing.T) {
	docID := uuid.New()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/index" || r.Header.Get("Authorization") != "Bearer test-service-token" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"schema_version":"index.v1","document_id":"` + docID.String() + `","outcome":"READY","error_code":null,"embedding":[1]}`))
	}))
	defer srv.Close()
	_, err := (HTTPWorker{BaseURL: srv.URL, Token: "test-service-token"}).Index(context.Background(), Request{
		SchemaVersion: SchemaVersion, DocumentID: docID, XAID: "xa_chu_se", DomainID: "ho_tich_chung_thuc",
		Checksum: "ab", ProcedureVersionID: uuid.New(), RelationshipType: "SOURCE",
	})
	if err == nil {
		t.Fatal("unknown embedding field was accepted")
	}
}

func TestDecodeStrictRejectsTrailingJSON(t *testing.T) {
	for _, raw := range []string{
		`{"schema_version":"index.v1"}{"extra":true}`,
		`{"schema_version":"index.v1"} true`,
		`{"schema_version":"index.v1"}}`,
		`[1]]`,
	} {
		var dest any
		if err := DecodeStrict(strings.NewReader(raw), &dest); err == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
	var dest map[string]any
	if err := DecodeStrict(strings.NewReader(`{"schema_version":"index.v1"}`), &dest); err != nil {
		t.Fatal(err)
	}
}

func TestHTTPWorkerHonorsConfiguredTimeoutAboveFiveSeconds(t *testing.T) {
	docID := uuid.New()
	versionID := uuid.New()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(6 * time.Second)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"schema_version":"index.v1","document_id":"` + docID.String() + `","procedure_version_id":"` + versionID.String() + `","outcome":"READY"}`))
	}))
	defer srv.Close()
	started := time.Now()
	res, err := (HTTPWorker{BaseURL: srv.URL, Token: "test-service-token", Timeout: 8 * time.Second}).Index(context.Background(), Request{
		SchemaVersion: SchemaVersion, DocumentID: docID, XAID: "xa_chu_se", DomainID: "ho_tich_chung_thuc",
		Checksum: strings.Repeat("a", 64), ProcedureVersionID: versionID, RelationshipType: "SOURCE",
	})
	if err != nil {
		t.Fatal(err)
	}
	if time.Since(started) < 6*time.Second {
		t.Fatal("worker returned before the handler finished")
	}
	if res.Outcome != OutcomeReady || res.DocumentID != docID || res.ProcedureVersionID != versionID {
		t.Fatalf("%+v", res)
	}
}
