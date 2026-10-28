package drafting

import (
	"strconv"
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

// joinOverlapping joins two consecutive chunks. Chunks overlap (the next one
// repeats the end of the previous one), so the longest suffix of a that is a
// prefix of b is written once.
func joinOverlapping(a, b string) string {
	for k := min(len(a), len(b)); k > 0; k-- {
		if strings.HasSuffix(a, b[:k]) {
			return a + b[k:]
		}
	}
	return a + " " + b
}

// excerptIndex finds the chunks next to an excerpt (same document and
// generation), so a quotation that runs across a chunk boundary still verifies.
type excerptIndex map[string]Excerpt

func newExcerptIndex(excerpts []Excerpt) excerptIndex {
	idx := excerptIndex{}
	for _, ex := range excerpts {
		idx[neighbourKey(ex, ex.ChunkIndex)] = ex
	}
	return idx
}

func neighbourKey(ex Excerpt, chunk int) string {
	return ex.DocumentID.String() + "|" + ex.GenerationID.String() + "|" + strconv.Itoa(chunk)
}

// quoteFound: the quotation is in the excerpt, or runs from it into the next
// chunk, or from the previous chunk into it. Only direct neighbours count; the
// page must lie in the joined range.
func (idx excerptIndex) quoteFound(quote string, page int, ex Excerpt) bool {
	if page >= ex.PageStart && page <= ex.PageEnd && quotationInText(quote, ex.Content) {
		return true
	}
	needle := normalizeQuote(quote)
	if needle == "" {
		return false
	}
	try := func(first, second Excerpt) bool {
		if page < min(first.PageStart, second.PageStart) || page > max(first.PageEnd, second.PageEnd) {
			return false
		}
		joined := joinOverlapping(normalizeQuote(first.Content), normalizeQuote(second.Content))
		return strings.Contains(joined, needle)
	}
	if next, ok := idx[neighbourKey(ex, ex.ChunkIndex+1)]; ok && try(ex, next) {
		return true
	}
	if prev, ok := idx[neighbourKey(ex, ex.ChunkIndex-1)]; ok && try(prev, ex) {
		return true
	}
	return false
}
