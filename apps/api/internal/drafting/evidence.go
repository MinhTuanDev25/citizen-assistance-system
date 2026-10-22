package drafting

import (
	"strings"
	"unicode"
)

func normalizeQuote(s string) string {
	var b strings.Builder
	space := true
	for _, r := range strings.ToLower(strings.TrimSpace(s)) {
		if unicode.IsSpace(r) {
			if !space {
				b.WriteByte(' ')
				space = true
			}
			continue
		}
		space = false
		b.WriteRune(r)
	}
	return b.String()
}

func quotationInText(quote, source string) bool {
	needle := normalizeQuote(quote)
	if needle == "" {
		return false
	}
	return strings.Contains(normalizeQuote(source), needle)
}
