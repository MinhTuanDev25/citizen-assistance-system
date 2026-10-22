package drafting

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
)

func lockProcedureIdentity(raw []byte, expectedCode, expectedName string) (locked []byte, originalCode string, mismatch bool, err error) {
	var obj map[string]any
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, "", false, fmt.Errorf("drafting: definition is not JSON: %w", err)
	}
	if obj == nil {
		obj = map[string]any{}
	}
	originalCode, _ = obj["procedure_code"].(string)
	originalCode = strings.TrimSpace(originalCode)
	mismatch = originalCode != "" && originalCode != expectedCode
	if originalCode == "" {
		mismatch = true
	}
	obj["procedure_code"] = expectedCode
	if expectedName != "" {
		obj["name"] = expectedName
	}
	delete(obj, "support_map")
	delete(obj, "approved_support_map")
	out, err := json.Marshal(obj)
	if err != nil {
		return nil, originalCode, mismatch, err
	}
	if _, parseErr := decision.ParseDefinition(out); parseErr != nil && !mismatch {
		return out, originalCode, mismatch, parseErr
	}
	return out, originalCode, mismatch, nil
}

func applyIdentityFinding(v *decision.DraftValidation, originalCode, expectedCode string, mismatch bool) {
	v.IdentityMismatch = mismatch
	if !mismatch {
		return
	}
	v.SchemaValid = false
	v.Readiness = "invalid"
	msg := "model or editor procedure_code does not match the locked target"
	if originalCode == "" {
		msg = "procedure_code was missing from the original definition"
	}
	v.Issues = append(v.Issues, decision.DraftIssue{
		Code:     "PROCEDURE_CODE_LOCKED",
		Severity: "error",
		Message:  fmt.Sprintf("%s (original=%q target=%q)", msg, originalCode, expectedCode),
	})
}
