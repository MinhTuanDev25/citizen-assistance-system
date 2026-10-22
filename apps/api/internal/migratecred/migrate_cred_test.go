package migratecred_test

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestProductionMigrationsHaveNoDemoCredentials(t *testing.T) {
	root := filepath.Join("..", "..", "..", "..", "deploy", "migrations")
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	bcrypt := regexp.MustCompile(`\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}`)
	quotedDemo := regexp.MustCompile(`'(admin123|citizen123)'`)

	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".sql") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(root, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		text := string(raw)
		if bcrypt.MatchString(text) {
			t.Errorf("%s contains bcrypt password hash", e.Name())
		}
		if quotedDemo.MatchString(text) {
			t.Errorf("%s contains quoted demo password literal", e.Name())
		}
		if strings.HasPrefix(e.Name(), "000004") {
			if !strings.Contains(text, "NULL") {
				t.Errorf("000004 must set password_hash NULL")
			}
			if regexp.MustCompile(`(?i)password_hash\s*=\s*'[^']+'`).MatchString(text) {
				t.Errorf("000004 must not assign a password_hash string")
			}
		}
	}
}
