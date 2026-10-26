package retrieve

import (
	_ "embed"
	"encoding/json"
	"strings"
)

//go:embed support_map.json
var supportMapJSON []byte

type supportMapFile struct {
	Schema  string         `json:"schema"`
	Entries []supportEntry `json:"entries"`
}

type supportEntry struct {
	XaID          string `json:"xa_id"`
	ProcedureCode string `json:"procedure_code"`
	DocumentID    string `json:"document_id"`
	GenerationID  string `json:"generation_id"`
	PageStart     int    `json:"page_start"`
	PageEnd       int    `json:"page_end"`
	TextSHA256    string `json:"text_sha256"`
	Statement     string `json:"statement"`
}

var compiledSupport []supportEntry

func init() {
	var file supportMapFile
	if err := json.Unmarshal(supportMapJSON, &file); err != nil {
		panic("retrieve: support_map.json: " + err.Error())
	}
	if file.Schema != "cas.verified-support.v1" {
		panic("retrieve: unsupported support map schema")
	}
	compiledSupport = file.Entries
}

// ClassifySupport labels a hit supporting only when it matches the verified
// claim map. Token/substring overlap is never enough.
func ClassifySupport(hit Hit, statements []string, procedureCode string) string {
	if hit.DocumentID == "" || hit.GenerationID == "" || hit.ContentSHA256 == "" {
		return "related"
	}
	if hit.PageStart < 1 || hit.PageEnd < hit.PageStart {
		return "related"
	}
	want := map[string]struct{}{}
	for _, item := range statements {
		item = strings.TrimSpace(item)
		if item != "" {
			want[item] = struct{}{}
		}
	}
	if len(want) == 0 {
		return "related"
	}
	code := strings.TrimSpace(procedureCode)
	for _, entry := range compiledSupport {
		if _, ok := want[entry.Statement]; !ok {
			continue
		}
		if code != "" && entry.ProcedureCode != code {
			continue
		}
		if entry.DocumentID != hit.DocumentID || entry.GenerationID != hit.GenerationID {
			continue
		}
		if entry.PageStart != hit.PageStart || entry.PageEnd != hit.PageEnd {
			continue
		}
		if !strings.EqualFold(entry.TextSHA256, hit.ContentSHA256) {
			continue
		}
		return "supporting"
	}
	return "related"
}
