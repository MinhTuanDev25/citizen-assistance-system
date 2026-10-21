package aiextract

import (
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// Grounding is a per-slot policy filter applied after contract/type checks.
// Failure drops that slot only — it is not a contract violation.
// Confidence is never used as proof that a value is true.

type dropReason string

const (
	dropEmptyEvidence     dropReason = "empty_evidence"
	dropInventedEvidence  dropReason = "invented_evidence"
	dropValueNotInMessage dropReason = "value_not_in_message"
	dropNoBooleanCues     dropReason = "no_boolean_cues"
	dropPolarityConflict  dropReason = "polarity_conflict"
	dropPolarityMismatch  dropReason = "polarity_mismatch"
	dropNegatedCue        dropReason = "negated_cue"
	dropUncertain         dropReason = "uncertain_clause"
	dropPreference        dropReason = "preference_not_fact"
	dropAlternative       dropReason = "alternative_disjunction"
	dropNoEnumCue         dropReason = "no_enum_cue"
	dropUnknownBoolean    dropReason = "unknown_boolean_without_cues"
)

type booleanCues struct {
	positive []string // already Fold-normalized
	negative []string
}

type enumCues map[string][]string // enum value -> folded phrases

// Branch-sensitive booleans and registrant enums for the thesis procedures.
// Unknown boolean keys are dropped rather than guessed.
var booleanCueTable = map[string]booleanCues{
	"da_ket_hon": {
		positive: []string{
			"cha me da dang ky ket hon",
			"da dang ky ket hon",
			"co dang ky ket hon",
			"cha me da ket hon",
			"da ket hon",
			"da cuoi",
		},
		negative: []string{
			"cha me chua dang ky ket hon",
			"cha me khong co dang ky ket hon",
			"khong co dang ky ket hon",
			"chua dang ky ket hon",
			"khong dang ky ket hon",
			"cha me chua ket hon",
			"chua ket hon",
			"khong ket hon",
			"chua cuoi",
		},
	},
	"co_giay_chung_sinh": {
		positive: []string{
			"da co giay chung sinh",
			"co giay chung sinh",
		},
		negative: []string{
			"chua co giay chung sinh",
			"khong co giay chung sinh",
			"chua co giay",
		},
	},
}

var enumCueTable = map[string]enumCues{
	"nguoi_di_dang_ky": {
		"me": {
			"me se di dang ky",
			"me di dang ky",
			"me se di",
			"me di",
			"me len xa",
		},
		"cha": {
			"cha se di dang ky",
			"bo se di dang ky",
			"cha di dang ky",
			"bo di dang ky",
			"cha se di",
			"bo se di",
			"cha di",
			"bo di",
		},
		"ong_ba": {
			"ong ba se di",
			"ong ba di",
			"ong ba",
		},
		"nguoi_duoc_uy_quyen": {
			"nguoi duoc uy quyen",
			"duoc uy quyen",
			"uy quyen",
		},
	},
}

var vnFold = map[rune]rune{
	'à': 'a', 'á': 'a', 'ạ': 'a', 'ả': 'a', 'ã': 'a',
	'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ậ': 'a', 'ẩ': 'a', 'ẫ': 'a',
	'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ặ': 'a', 'ẳ': 'a', 'ẵ': 'a',
	'è': 'e', 'é': 'e', 'ẹ': 'e', 'ẻ': 'e', 'ẽ': 'e',
	'ê': 'e', 'ề': 'e', 'ế': 'e', 'ệ': 'e', 'ể': 'e', 'ễ': 'e',
	'ì': 'i', 'í': 'i', 'ị': 'i', 'ỉ': 'i', 'ĩ': 'i',
	'ò': 'o', 'ó': 'o', 'ọ': 'o', 'ỏ': 'o', 'õ': 'o',
	'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ộ': 'o', 'ổ': 'o', 'ỗ': 'o',
	'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ợ': 'o', 'ở': 'o', 'ỡ': 'o',
	'ù': 'u', 'ú': 'u', 'ụ': 'u', 'ủ': 'u', 'ũ': 'u',
	'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ự': 'u', 'ử': 'u', 'ữ': 'u',
	'ỳ': 'y', 'ý': 'y', 'ỵ': 'y', 'ỷ': 'y', 'ỹ': 'y',
	'đ': 'd',
}

// foldMessage NFC-normalizes, lowercases, strips Vietnamese diacritics, and
// collapses non-alnum to single spaces. Same family as decision.Fold.
func foldMessage(s string) string {
	s = norm.NFC.String(s)
	var b strings.Builder
	b.Grow(len(s))
	prevSpace := true
	for _, r := range strings.ToLower(s) {
		if mapped, ok := vnFold[r]; ok {
			r = mapped
		}
		if unicode.IsDigit(r) || (r >= 'a' && r <= 'z') {
			b.WriteRune(r)
			prevSpace = false
			continue
		}
		if !prevSpace {
			b.WriteByte(' ')
			prevSpace = true
		}
	}
	return strings.TrimSpace(b.String())
}

func containsFoldedPhrase(folded, phrase string) bool {
	if phrase == "" || folded == "" {
		return false
	}
	return strings.Contains(" "+folded+" ", " "+phrase+" ")
}

func containsFoldedSpan(folded, span string) bool {
	if span == "" || folded == "" {
		return false
	}
	if containsFoldedPhrase(folded, span) {
		return true
	}
	return strings.Contains(folded, span)
}

// GroundSlot reports whether a contract-valid proposal is supported by the
// citizen message. Invented evidence spans and unsupported inferences fail.
func GroundSlot(message string, spec SlotSpec, key string, value any, evidence string) (bool, dropReason) {
	foldedMsg := foldMessage(message)
	foldedEv := foldMessage(evidence)
	if foldedEv == "" {
		return false, dropEmptyEvidence
	}
	if !containsFoldedSpan(foldedMsg, foldedEv) {
		return false, dropInventedEvidence
	}
	switch spec.Type {
	case "boolean":
		b, ok := value.(bool)
		if !ok {
			return false, dropPolarityMismatch
		}
		return groundBoolean(message, key, b)
	case "enum":
		s, _ := value.(string)
		return groundEnum(message, key, s)
	case "string":
		s, _ := value.(string)
		foldedVal := foldMessage(s)
		if foldedVal == "" || !containsFoldedSpan(foldedMsg, foldedVal) {
			return false, dropValueNotInMessage
		}
		return true, ""
	case "number":
		return groundNumber(foldedMsg, value)
	default:
		return false, dropValueNotInMessage
	}
}

func groundBoolean(message, key string, proposed bool) (bool, dropReason) {
	cues, ok := booleanCueTable[key]
	if !ok {
		return false, dropUnknownBoolean
	}
	posHit, negHit, uncertain := false, false, false
	prevTouched := false
	for _, clause := range splitClauses(message) {
		folded := foldMessage(clause)
		flags := classifyClause(folded)
		touched := clauseTouchesCues(folded, cues.positive) || clauseTouchesCues(folded, cues.negative)
		if flags.uncertain && !touched && prevTouched {
			uncertain = true
		}
		if touched && flags.uncertain {
			uncertain = true
			prevTouched = touched
			continue
		}
		if hit, negated := longestCueHit(folded, cues.positive); hit {
			if negated {
				negHit = true
			} else {
				posHit = true
			}
		}
		if hit, negated := longestCueHit(folded, cues.negative); hit {
			if negated {
				posHit = true
				negHit = true
			} else {
				negHit = true
			}
		}
		prevTouched = touched
	}
	if uncertain {
		return false, dropUncertain
	}
	if posHit && negHit {
		return false, dropPolarityConflict
	}
	if !posHit && !negHit {
		return false, dropNoBooleanCues
	}
	if proposed && posHit {
		return true, ""
	}
	if proposed && negHit {
		return false, dropNegatedCue
	}
	if !proposed && negHit {
		return true, ""
	}
	return false, dropPolarityMismatch
}

func groundEnum(message, key, value string) (bool, dropReason) {
	table, ok := enumCueTable[key]
	var phrases []string
	if ok {
		phrases = table[value]
	} else {
		foldedVal := foldMessage(strings.ReplaceAll(value, "_", " "))
		if foldedVal != "" {
			phrases = []string{foldedVal}
		}
	}
	if len(phrases) == 0 {
		return false, dropNoEnumCue
	}
	positive, negatedOnly := false, false
	prevTouched := false
	for _, clause := range splitClauses(message) {
		folded := foldMessage(clause)
		flags := classifyClause(folded)
		hit, negated := longestCueHit(folded, phrases)
		anyValue := clauseTouchesEnumKey(folded, key)
		if flags.uncertain && !anyValue && prevTouched {
			return false, dropUncertain
		}
		if (hit || anyValue) && flags.uncertain {
			return false, dropUncertain
		}
		if (hit || anyValue) && flags.preference {
			return false, dropPreference
		}
		if flags.alternative && enumDisjunction(folded, key) {
			return false, dropAlternative
		}
		prevTouched = anyValue
		if !hit {
			continue
		}
		if negated {
			negatedOnly = true
			continue
		}
		positive = true
	}
	if positive {
		return true, ""
	}
	if negatedOnly {
		return false, dropNegatedCue
	}
	return false, dropNoEnumCue
}

func groundNumber(foldedMsg string, value any) (bool, dropReason) {
	var n string
	switch v := value.(type) {
	case float64:
		n = strconv.FormatFloat(v, 'f', -1, 64)
	case int:
		n = strconv.Itoa(v)
	default:
		return false, dropValueNotInMessage
	}
	if containsFoldedPhrase(foldedMsg, n) || strings.Contains(foldedMsg, n) {
		return true, ""
	}
	return false, dropValueNotInMessage
}

var (
	clauseSplitRE = regexp.MustCompile(`(?i)[,.;!?]+|\s+mà\s+|\s+nhưng\s+`)
	// Longest first. These mark the start of a negated VP in a folded clause.
	negationPrefixes = []string{
		"khong phai",
		"khong con",
		"khong co",
		"chang phai",
		"chang",
		"chua",
		"khong",
	}
	// Clause-local epistemic hedges. Longest first. Bare "chua"/"khong" stay
	// in negationPrefixes so factual negatives are not globally rejected.
	uncertaintyPhrases = []string{
		"khong xac dinh duoc",
		"khong biet",
		"khong ro",
		"chua ro",
		"chua biet",
		"chua chac",
		"khong chac",
		"khong nho",
		"hinh nhu",
		"co le",
		"chac la",
		"co the se",
		"co the",
	}
	preferencePhrases = []string{
		"khong muon",
		"du dinh",
		"muon",
	}
	alternativePhrases = []string{"hay", "hoac"}
)

type clauseFlags struct {
	uncertain   bool
	preference  bool
	alternative bool
}

func classifyClause(folded string) clauseFlags {
	return clauseFlags{
		uncertain:   hasAnyPhrase(folded, uncertaintyPhrases),
		preference:  hasAnyPhrase(folded, preferencePhrases),
		alternative: hasAnyPhrase(folded, alternativePhrases),
	}
}

func hasAnyPhrase(folded string, phrases []string) bool {
	for _, p := range phrases {
		if phraseIndex(folded, p) >= 0 {
			return true
		}
	}
	return false
}

func clauseTouchesCues(folded string, cues []string) bool {
	hit, _ := longestCueHit(folded, cues)
	return hit
}

func clauseTouchesEnumKey(folded, key string) bool {
	table, ok := enumCueTable[key]
	if !ok {
		return false
	}
	for _, phrases := range table {
		if clauseTouchesCues(folded, phrases) {
			return true
		}
	}
	return false
}

func enumDisjunction(folded, key string) bool {
	table, ok := enumCueTable[key]
	if !ok {
		return false
	}
	hits := 0
	for _, phrases := range table {
		if clauseTouchesCues(folded, phrases) {
			hits++
		}
	}
	return hits >= 2
}

func splitClauses(message string) []string {
	parts := clauseSplitRE.Split(norm.NFC.String(message), -1)
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		return []string{message}
	}
	return out
}

func longestCueHit(folded string, cues []string) (found bool, negated bool) {
	ordered := append([]string(nil), cues...)
	sort.Slice(ordered, func(i, j int) bool { return len(ordered[i]) > len(ordered[j]) })
	for _, cue := range ordered {
		if cue == "" {
			continue
		}
		idx := phraseIndex(folded, cue)
		if idx < 0 {
			continue
		}
		return true, isLocallyNegated(folded, idx)
	}
	return false, false
}

func phraseIndex(folded, phrase string) int {
	if phrase == "" || folded == "" {
		return -1
	}
	padded := " " + folded + " "
	needle := " " + phrase + " "
	pos := strings.Index(padded, needle)
	if pos < 0 {
		return -1
	}
	return pos
}

func isLocallyNegated(folded string, cueStart int) bool {
	if cueStart < 0 || cueStart > len(folded) {
		return false
	}
	prefix := strings.TrimSpace(folded[:cueStart])
	if prefix == "" {
		return false
	}
	for _, neg := range negationPrefixes {
		if prefix == neg || strings.HasSuffix(prefix, " "+neg) {
			return true
		}
	}
	return false
}
