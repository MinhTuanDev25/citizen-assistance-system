package docs_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

const (
	pathMessagesPost  = "/api/v1/sessions/{sessionId}/messages"
	pathTurnsPost     = "/api/v1/sessions/{sessionId}/turns"
	pathByCodeGet     = "/api/v1/procedures/by-code/{code}"
	pathActiveVersion = "/api/v1/procedures/{id}/active-version"
)

type swaggerDoc struct {
	Paths map[string]pathMethods `json:"paths" yaml:"paths"`
}

type pathMethods struct {
	Get  *operation `json:"get" yaml:"get"`
	Post *operation `json:"post" yaml:"post"`
}

type operation struct {
	Summary     string              `json:"summary" yaml:"summary"`
	Description string              `json:"description" yaml:"description"`
	Parameters  []parameter         `json:"parameters" yaml:"parameters"`
	Responses   map[string]response `json:"responses" yaml:"responses"`
}

type parameter struct {
	Name        string `json:"name" yaml:"name"`
	In          string `json:"in" yaml:"in"`
	Required    bool   `json:"required" yaml:"required"`
	Description string `json:"description" yaml:"description"`
}

type response struct {
	Description string `json:"description" yaml:"description"`
}

func docsRoot(t *testing.T) string {
	t.Helper()
	candidates := []string{".", filepath.Join("..", "docs")}
	for _, root := range candidates {
		if _, err := os.Stat(filepath.Join(root, "swagger.json")); err == nil {
			return root
		}
	}
	t.Fatal("swagger.json not found relative to test cwd")
	return ""
}

func TestOpenAPIMatchesHandlers(t *testing.T) {
	root := docsRoot(t)
	jsonPath := filepath.Join(root, "swagger.json")
	yamlPath := filepath.Join(root, "swagger.yaml")
	docsGoPath := filepath.Join(root, "docs.go")

	for _, p := range []string{jsonPath, yamlPath, docsGoPath} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("expected doc artifact %s: %v", p, err)
		}
	}

	jsonBytes, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatal(err)
	}
	var doc swaggerDoc
	if err := json.Unmarshal(jsonBytes, &doc); err != nil {
		t.Fatalf("parse swagger.json: %v", err)
	}
	assertMessagesPost(t, doc)
	assertTurnsPost(t, doc)
	assertProcedureDetailGet(t, doc, pathByCodeGet)
	assertProcedureDetailGet(t, doc, pathActiveVersion)

	yamlBytes, err := os.ReadFile(yamlPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(yamlBytes), "# CAS contract:") {
		t.Error("swagger.yaml must not contain CAS contract sentinel comments")
	}
	var yamlDoc swaggerDoc
	if err := yaml.Unmarshal(yamlBytes, &yamlDoc); err != nil {
		t.Fatalf("parse swagger.yaml: %v", err)
	}
	assertMessagesPost(t, yamlDoc)
	assertTurnsPost(t, yamlDoc)
	assertProcedureDetailGet(t, yamlDoc, pathByCodeGet)
	assertProcedureDetailGet(t, yamlDoc, pathActiveVersion)

	docsGo, err := os.ReadFile(docsGoPath)
	if err != nil {
		t.Fatal(err)
	}
	docsStr := string(docsGo)
	if !strings.Contains(docsStr, pathTurnsPost) {
		t.Errorf("docs.go embed missing turns path %q", pathTurnsPost)
	}
	if !strings.Contains(docsStr, "CONFIRM_INTENT") {
		t.Error("docs.go embed missing CONFIRM_INTENT in turn documentation")
	}
	for _, p := range []string{pathByCodeGet, pathActiveVersion} {
		if !strings.Contains(docsStr, p) {
			t.Errorf("docs.go embed missing procedure path %q", p)
		}
	}
	if !strings.Contains(docsStr, "CitizenDomainIDs") {
		t.Error("docs.go embed missing CitizenDomainIDs scope description")
	}
	if !strings.Contains(docsStr, "403") {
		t.Error("docs.go embed missing 403 responses for procedure detail")
	}
}

func assertMessagesPost(t *testing.T, doc swaggerDoc) {
	t.Helper()
	methods, ok := doc.Paths[pathMessagesPost]
	if !ok {
		t.Fatalf("paths missing %q", pathMessagesPost)
	}
	if methods.Post == nil {
		t.Fatalf("%s has no post operation", pathMessagesPost)
	}
	if _, has201 := methods.Post.Responses["201"]; has201 {
		t.Errorf("%s POST must not declare response 201", pathMessagesPost)
	}
	if _, has410 := methods.Post.Responses["410"]; !has410 {
		t.Errorf("%s POST must declare response 410", pathMessagesPost)
	}
}

func assertTurnsPost(t *testing.T, doc swaggerDoc) {
	t.Helper()
	methods, ok := doc.Paths[pathTurnsPost]
	if !ok {
		t.Fatalf("paths missing %q", pathTurnsPost)
	}
	if methods.Post == nil {
		t.Fatalf("%s has no post operation", pathTurnsPost)
	}
	op := methods.Post

	var xReqID *parameter
	for i := range op.Parameters {
		p := &op.Parameters[i]
		if p.Name == "X-Request-ID" && p.In == "header" {
			xReqID = p
			break
		}
	}
	if xReqID == nil {
		t.Fatalf("%s POST missing X-Request-ID header parameter", pathTurnsPost)
	}
	if !xReqID.Required {
		t.Errorf("%s X-Request-ID must be required", pathTurnsPost)
	}

	if _, has409 := op.Responses["409"]; !has409 {
		t.Errorf("%s POST must declare response 409", pathTurnsPost)
	}

	combined := op.Summary + " " + op.Description
	for _, p := range op.Parameters {
		combined += " " + p.Description
	}
	for code, r := range op.Responses {
		combined += " " + code + " " + r.Description
	}
	if !strings.Contains(combined, "CONFIRM_INTENT") {
		t.Errorf("%s POST operation text must mention CONFIRM_INTENT", pathTurnsPost)
	}
}

func assertProcedureDetailGet(t *testing.T, doc swaggerDoc, path string) {
	t.Helper()
	methods, ok := doc.Paths[path]
	if !ok {
		t.Fatalf("paths missing %q", path)
	}
	if methods.Get == nil {
		t.Fatalf("%s has no get operation", path)
	}
	op := methods.Get

	var auth *parameter
	for i := range op.Parameters {
		p := &op.Parameters[i]
		if strings.EqualFold(p.Name, "Authorization") && p.In == "header" {
			auth = p
			break
		}
	}
	if auth == nil {
		t.Fatalf("%s GET missing Authorization header parameter", path)
	}
	authDesc := strings.ToLower(auth.Description)
	if !strings.Contains(authDesc, "admin") || !strings.Contains(authDesc, "bearer") {
		t.Errorf("%s Authorization description must mention Bearer ADMIN: %q", path, auth.Description)
	}

	resp403, has403 := op.Responses["403"]
	if !has403 {
		t.Fatalf("%s GET must declare response 403", path)
	}

	combined := strings.ToLower(op.Summary + " " + op.Description + " " + auth.Description + " " + resp403.Description)
	for _, needle := range []string{"citizendomainids", "guest", "citizen", "admin"} {
		if !strings.Contains(combined, needle) {
			t.Errorf("%s GET docs must describe scope including %q; got: %s", path, needle, combined)
		}
	}
}
