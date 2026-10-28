//go:build integration

package activation_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"testing"

	"github.com/google/uuid"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
)

// TestCitationContractActivation checks how many citations a Definition can
// carry through version activation, persistence and the public API. The
// version/activation layer has no citation limit (manual versions may cite
// freely); drafted Definitions are capped by decision.MaxDraftCitations before
// approval, so a draft with 21 never reaches this layer.
func TestCitationContractActivation(t *testing.T) {
	pool := requireP4DB(t)
	ctx := context.Background()
	for _, n := range []int{10, 11, 15, 20, 21} {
		t.Run(fmt.Sprintf("%d_citations", n), func(t *testing.T) {
			fx := newFixture(t, pool, false)
			var def map[string]any
			var code string
			if err := pool.QueryRow(ctx, `SELECT p.procedure_code FROM procedures p WHERE p.id = $1`, fx.ProcedureID).Scan(&code); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(validDef(code, code, "2.0.0"), &def); err != nil {
				t.Fatal(err)
			}
			cites := make([]map[string]any, 0, n)
			for i := 0; i < n; i++ {
				cites = append(cites, map[string]any{"doc_id": fmt.Sprintf("doc-%d", i), "title": fmt.Sprintf("Văn bản %d", i), "source_type": "ai_draft_unapproved"})
			}
			def["citations"] = cites
			raw, _ := json.Marshal(def)
			if _, err := pool.Exec(ctx, `UPDATE procedure_versions SET definition = $2::jsonb WHERE id = $1`, fx.V2, raw); err != nil {
				t.Fatal(err)
			}

			// Draft-layer rule for the same Definition.
			tooMany := false
			for _, is := range decision.ValidateDraftDefinition(raw, code).Issues {
				tooMany = tooMany || is.Code == "TOO_MANY_CITATIONS"
			}
			if tooMany != (n > decision.MaxDraftCitations) {
				t.Fatalf("draft validator TOO_MANY_CITATIONS=%v for %d", tooMany, n)
			}

			out, err := fx.svc.Activate(ctx, actor(), uuid.New(), fx.ProcedureID, fx.V2, nil)
			if err != nil || out.Status != "ACTIVE" {
				t.Fatalf("activate %d citations: %+v err=%v", n, out, err)
			}
			var stored []byte
			if err := pool.QueryRow(ctx, `SELECT definition FROM procedure_versions WHERE id = $1`, fx.V2).Scan(&stored); err != nil {
				t.Fatal(err)
			}
			parsed, err := decision.ParseDefinition(stored)
			if err != nil || len(parsed.Citations) != n {
				t.Fatalf("stored definition: err=%v citations=%d", err, len(parsed.Citations))
			}

			eng, _, _ := testEngine(t, pool)
			status, env := callJSON(t, eng, http.MethodGet, fmt.Sprintf("/api/v1/procedures/%s/active-version", fx.ProcedureID), nil, "", uuid.Nil)
			if status != http.StatusOK {
				t.Fatalf("public %d %v", status, env)
			}
			data, _ := env["data"].(map[string]any)
			body, _ := json.Marshal(data)
			var served struct {
				Definition struct {
					Citations []map[string]any `json:"citations"`
				} `json:"definition"`
			}
			_ = json.Unmarshal(body, &served)
			if len(served.Definition.Citations) != n {
				t.Fatalf("public API served %d of %d citations", len(served.Definition.Citations), n)
			}
		})
	}
}
