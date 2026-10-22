package document

import (
	"path"
	"strings"
	"time"
	"unicode/utf8"
)

const maxMetaRunes = 200

func cleanFilename(name string) error {
	if name == "" || strings.ContainsRune(name, 0) {
		return errValidation("filename is not allowed")
	}
	if strings.Contains(name, "/") || strings.Contains(name, `\`) || strings.Contains(name, "..") {
		return errValidation("filename is not allowed")
	}
	base := path.Base(name)
	if base == "." || base == ".." || base != name {
		return errValidation("filename is not allowed")
	}
	if !strings.HasSuffix(strings.ToLower(base), ".pdf") {
		return errValidation("only .pdf files are accepted")
	}
	if utf8.RuneCountInString(base) > maxMetaRunes {
		return errValidation("filename is not allowed")
	}
	return nil
}

func cleanMeta(s string, required bool) (string, error) {
	s = strings.TrimSpace(s)
	if strings.ContainsRune(s, 0) {
		return "", errValidation("metadata is not allowed")
	}
	if required && s == "" {
		return "", errValidation("a required field is missing")
	}
	if utf8.RuneCountInString(s) > maxMetaRunes {
		return "", errValidation("metadata is too long")
	}
	return s, nil
}

func cleanDomainID(id string) (string, error) {
	id = strings.TrimSpace(id)
	if id == "" || len(id) > 64 {
		return "", errValidation("domain is not available")
	}
	for _, r := range id {
		if (r < 'a' || r > 'z') && (r < '0' || r > '9') && r != '_' {
			return "", errValidation("domain is not available")
		}
	}
	return id, nil
}

func cleanDate(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	parsed, err := time.Parse("2006-01-02", s)
	if err != nil || parsed.Format("2006-01-02") != s {
		return "", errValidation("date must be a real YYYY-MM-DD date")
	}
	return s, nil
}

func allowedMIME(v string) bool {
	v = strings.ToLower(strings.TrimSpace(v))
	if i := strings.Index(v, ";"); i >= 0 {
		v = strings.TrimSpace(v[:i])
	}
	return v == MimePDF
}
