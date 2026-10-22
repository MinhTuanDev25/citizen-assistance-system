package documentapi

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type formParam struct {
	Name     string `json:"name"`
	In       string `json:"in"`
	Type     string `json:"type"`
	Required bool   `json:"required"`
}

func TestAdminDocumentOpenAPIContract(t *testing.T) {
	path := filepath.Join("..", "..", "..", "..", "..", "docs", "swagger.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var spec struct {
		Paths map[string]map[string]struct {
			Security   []map[string][]string      `json:"security"`
			Consumes   []string                   `json:"consumes"`
			Parameters []formParam                `json:"parameters"`
			Responses  map[string]json.RawMessage `json:"responses"`
		} `json:"paths"`
		SecurityDefinitions map[string]struct {
			Type string `json:"type"`
			Name string `json:"name"`
			In   string `json:"in"`
		} `json:"securityDefinitions"`
	}
	if err := json.Unmarshal(raw, &spec); err != nil {
		t.Fatal(err)
	}
	auth, ok := spec.SecurityDefinitions["BearerAuth"]
	if !ok || auth.Type != "apiKey" || auth.Name != "Authorization" || auth.In != "header" {
		t.Fatal("BearerAuth security definition missing")
	}
	checks := []struct {
		path   string
		method string
		codes  []string
		form   []string
	}{
		{"/api/v1/admin/documents", "post", []string{"201", "401", "403", "409", "413", "422", "500"}, []string{"file", "title", "domain_id", "document_number", "issuer", "effective_date", "expire_date", "issued_date"}},
		{"/api/v1/admin/documents", "get", []string{"200", "401", "403", "422", "500"}, nil},
		{"/api/v1/admin/documents/{id}", "get", []string{"200", "401", "403", "404", "500"}, nil},
		{"/api/v1/admin/documents/{id}/content", "get", []string{"200", "401", "403", "404", "500"}, nil},
		{"/api/v1/admin/documents/link-targets", "get", []string{"200", "401", "403", "500"}, nil},
		{"/api/v1/admin/documents/{id}/links", "get", []string{"200", "401", "403", "404", "500"}, nil},
		{"/api/v1/admin/documents/{id}/links", "post", []string{"200", "201", "400", "401", "403", "404", "409", "422", "500"}, nil},
		{"/api/v1/admin/documents/{id}/links/{versionId}", "delete", []string{"200", "400", "401", "403", "404", "409", "422", "500"}, nil},
		{"/api/v1/admin/documents/{id}/links/{versionId}/index", "post", []string{"200", "400", "401", "403", "404", "409", "422", "500"}, nil},
		{"/api/v1/admin/documents/{id}/links/{versionId}/index/retry", "post", []string{"200", "400", "401", "403", "404", "409", "422", "500"}, nil},
		{"/api/v1/admin/documents/{id}/links/{versionId}/index/reindex", "post", []string{"200", "400", "401", "403", "404", "409", "422", "500"}, nil},
		{"/api/v1/admin/documents/index-metrics", "get", []string{"200", "401", "403"}, nil},
	}
	for _, check := range checks {
		item, ok := spec.Paths[check.path]
		if !ok {
			t.Fatalf("missing path %s", check.path)
		}
		op, ok := item[check.method]
		if !ok {
			t.Fatalf("missing %s %s", check.method, check.path)
		}
		if !hasBearer(op.Security) {
			t.Fatalf("%s %s missing BearerAuth", check.method, check.path)
		}
		for _, code := range check.codes {
			if _, ok := op.Responses[code]; !ok {
				t.Fatalf("%s %s missing response %s", check.method, check.path, code)
			}
		}
		if check.method == "post" && len(check.form) > 0 {
			joined := ""
			for _, c := range op.Consumes {
				joined += c + " "
			}
			if !strings.Contains(joined, "multipart/form-data") {
				t.Fatalf("upload consumes %v", op.Consumes)
			}
			for _, name := range check.form {
				if !hasFormField(op.Parameters, name) {
					t.Fatalf("upload missing form field %s", name)
				}
			}
		}
	}
}

func hasBearer(security []map[string][]string) bool {
	for _, item := range security {
		if _, ok := item["BearerAuth"]; ok {
			return true
		}
	}
	return false
}

func hasFormField(params []formParam, name string) bool {
	for _, p := range params {
		if p.Name == name && p.In == "formData" {
			if name == "file" && (p.Type != "file" || !p.Required) {
				return false
			}
			if (name == "title" || name == "domain_id") && !p.Required {
				return false
			}
			return true
		}
	}
	return false
}
