package drafting

import (
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

// A Vietnamese administrative-procedure decision lists many procedures. Part II
// gives each one a numbered heading followed by "Trình tự thực hiện", e.g.
// "1.Thủ tục đăng ký khai sinh Trình tự thực hiện:". Chunks overlap, so a heading
// can appear in two consecutive chunks; the first occurrence starts the section.
var sectionHeadingRe = regexp.MustCompile(`(?i)(?:^|[^0-9.,])([0-9]{1,2})\s*\.\s*(thủ\s+tục\s[^:]{3,200}?)\s*trình\s+tự`)

// Section is one procedure section of a long document, as a contiguous chunk range.
type Section struct {
	Index      int    `json:"index"`
	Number     int    `json:"number"`
	Title      string `json:"title"`
	ChunkStart int    `json:"chunk_start"`
	ChunkEnd   int    `json:"chunk_end"`
	PageStart  int    `json:"page_start"`
	PageEnd    int    `json:"page_end"`
	Chunks     int    `json:"chunks"`
	Chars      int    `json:"chars"`
	// BoundaryUncertain: the next heading number was not the expected one, so an
	// undetected (badly OCR'd) heading may be inside this section.
	BoundaryUncertain bool    `json:"boundary_uncertain"`
	Match             float64 `json:"match"`
	ExactMatch        bool    `json:"exact_match"`
}

type headingHit struct {
	number int
	title  string
	chunk  int
}

// DetectSections splits ordered excerpts into procedure sections. Text before the
// first heading (the summary table of Part I) belongs to no section.
func DetectSections(excerpts []Excerpt) []Section {
	hits := make([]headingHit, 0)
	// lastChunk[key]: chunk where this heading was last seen. Only a repeat in
	// the same or the next chunk is the overlap copy; a later repeat (e.g. an
	// appendix restating "1.Thủ tục đăng ký khai sinh") is a separate section.
	lastChunk := map[string]int{}
	for _, ex := range excerpts {
		for _, m := range sectionHeadingRe.FindAllStringSubmatch(ex.Content, -1) {
			n, err := strconv.Atoi(m[1])
			if err != nil {
				continue
			}
			title := cleanHeading(m[2])
			key := strconv.Itoa(n) + "|" + normalizeTitle(title)
			last, ok := lastChunk[key]
			lastChunk[key] = ex.ChunkIndex
			if ok && ex.ChunkIndex-last <= 1 {
				continue
			}
			hits = append(hits, headingHit{number: n, title: title, chunk: ex.ChunkIndex})
		}
	}
	if len(hits) == 0 || len(excerpts) == 0 {
		return []Section{}
	}
	finalChunk := excerpts[len(excerpts)-1].ChunkIndex
	out := make([]Section, 0, len(hits))
	for i, h := range hits {
		end := finalChunk
		if i+1 < len(hits) {
			end = hits[i+1].chunk - 1
			if end < h.chunk {
				end = h.chunk
			}
		}
		sec := Section{Index: i, Number: h.number, Title: h.title, ChunkStart: h.chunk, ChunkEnd: end}
		if i+1 < len(hits) {
			next := hits[i+1].number
			sec.BoundaryUncertain = next != h.number+1 && next != 1
		}
		for _, ex := range excerpts {
			if ex.ChunkIndex < sec.ChunkStart || ex.ChunkIndex > sec.ChunkEnd {
				continue
			}
			if sec.PageStart == 0 || ex.PageStart < sec.PageStart {
				sec.PageStart = ex.PageStart
			}
			if ex.PageEnd > sec.PageEnd {
				sec.PageEnd = ex.PageEnd
			}
			sec.Chunks++
			sec.Chars += len([]rune(ex.Content))
		}
		out = append(out, sec)
	}
	return out
}

// ScoreSections rates every section title against the procedure name. Only an
// exact (normalised) title match is ever auto-selected; similar titles such as
// "đăng ký lại khai sinh" or "... có yếu tố nước ngoài" are different procedures.
func ScoreSections(sections []Section, procedureName string) (best int) {
	best = -1
	want := normalizeTitle(procedureName)
	exact := 0
	// Score every section first, so all exact matches are marked even when
	// there are several; only then decide.
	for i := range sections {
		got := normalizeTitle(sections[i].Title)
		sections[i].ExactMatch = want != "" && got == want
		sections[i].Match = titleSimilarity(want, got)
		if sections[i].ExactMatch {
			exact++
			best = i
		}
	}
	if exact > 1 {
		// Several sections with the same title: never guess which one is meant.
		return -2
	}
	return best
}

func cleanHeading(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// normalizeTitle lowercases, drops the leading "thủ tục" and all punctuation.
func normalizeTitle(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		} else {
			b.WriteRune(' ')
		}
	}
	words := strings.Fields(b.String())
	if len(words) >= 2 && words[0] == "thủ" && words[1] == "tục" {
		words = words[2:]
	}
	return strings.Join(words, " ")
}

// titleSimilarity is the Dice coefficient over words; used only to rank choices.
func titleSimilarity(a, b string) float64 {
	wa, wb := strings.Fields(a), strings.Fields(b)
	if len(wa) == 0 || len(wb) == 0 {
		return 0
	}
	count := map[string]int{}
	for _, w := range wa {
		count[w]++
	}
	common := 0
	for _, w := range wb {
		if count[w] > 0 {
			count[w]--
			common++
		}
	}
	return float64(2*common) / float64(len(wa)+len(wb))
}
