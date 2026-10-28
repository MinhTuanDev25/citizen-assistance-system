package aiextract

import (
	"regexp"
	"slices"
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
	return GroundSlotFor("", message, spec, key, value, evidence)
}

// GroundSlotFor is GroundSlot with the procedure's name, whose words do not
// count as naming a question's topic.
func GroundSlotFor(procedureName, message string, spec SlotSpec, key string, value any, evidence string) (bool, dropReason) {
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
		if _, known := booleanCueTable[key]; !known {
			return groundQuestionBoolean(spec.Question, procedureName, foldedEv, b)
		}
		return groundBoolean(message, key, b)
	case "enum":
		s, _ := value.(string)
		return groundEnum(message, key, s, spec, foldedEv)
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

// questionFiller: words of a yes/no question that do not say what it is about.
var questionFiller = map[string]bool{
	"anh": true, "chi": true, "co": true, "khong": true, "hay": true, "da": true,
	"chua": true, "la": true, "phai": true, "cua": true, "cho": true, "duoc": true, "nay": true,
	"nao": true, "gi": true, "toi": true, "minh": true, "can": true, "muon": true, "hoac": true,
	"va": true, "voi": true, "thi": true, "se": true, "dang": true, "roi": true, "nguoi": true,
	"thuc": true, "hien": true, "thu": true, "tuc": true, "yeu": true, "cau": true,
}

var answerNegation = map[string]bool{"khong": true, "chua": true, "ko": true, "k": true, "chang": true}

// groundQuestionBoolean grounds a yes/no answer for a question without a
// hand-written cue table (Admin/AI-drafted procedures). The evidence span is
// verbatim in the message (checked before). It must name what the question is
// about: at least half of the question's distinctive words, i.e. its words
// minus filler and minus the procedure's own name ("bản sao", not "giấy khai
// sinh", for "Bạn có yêu cầu cấp bản sao Giấy khai sinh?"). Its polarity must
// match: a negation word (không, chưa) means false, none means true. Hedged
// spans ("không biết") are dropped.
func groundQuestionBoolean(question, procedureName, foldedEv string, proposed bool) (bool, dropReason) {
	words := strings.Fields(foldMessage(question))
	if len(words) > 0 && words[0] == "ban" { // "Bạn có …": the address, not "bản"
		words = words[1:]
	}
	generic := map[string]bool{}
	for _, w := range strings.Fields(foldMessage(procedureName)) {
		generic[w] = true
	}
	topic, distinct := map[string]bool{}, map[string]bool{}
	for _, w := range words {
		if questionFiller[w] || len(w) < 2 {
			continue
		}
		topic[w] = true
		if !generic[w] {
			distinct[w] = true
		}
	}
	if len(distinct) == 0 {
		distinct = topic
	}
	if len(distinct) == 0 {
		return false, dropUnknownBoolean
	}
	hits, negated := map[string]bool{}, false
	for _, w := range strings.Fields(foldedEv) {
		if distinct[w] {
			hits[w] = true
		}
		if answerNegation[w] {
			negated = true
		}
	}
	if len(hits) == 0 || len(hits)*2 < len(distinct) {
		return false, dropNoBooleanCues
	}
	if classifyClause(foldedEv).uncertain {
		return false, dropUncertain
	}
	if proposed == !negated {
		return true, ""
	}
	return false, dropPolarityMismatch
}

func groundEnum(message, key, value string, spec SlotSpec, foldedEv string) (bool, dropReason) {
	table, ok := enumCueTable[key]
	var phrases []string
	if ok {
		phrases = table[value]
	} else {
		phrases = enumPhrases(spec, value, foldedEv)
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

// enumPhrases: the cues of an enum value for a slot without a hand-written
// cue table (Admin/AI-drafted procedures): its code ("tre sinh thuong"), its
// label ("tre sinh binh thuong"), and the model's evidence span when that span
// clearly names this value and no other ("sinh thuong": its words belong to
// tre_sinh_thuong, and at least one of them to no other choice). The span is
// verbatim in the message (checked before), so this never accepts invented
// text; negation and hedging are still checked clause by clause.
func enumPhrases(spec SlotSpec, value, foldedEv string) []string {
	idx := -1
	for i, v := range spec.EnumValues {
		if v == value {
			idx = i
		}
	}
	words := func(i int) map[string]bool {
		out := map[string]bool{}
		text := strings.ReplaceAll(spec.EnumValues[i], "_", " ")
		if i < len(spec.EnumLabels) {
			text += " " + spec.EnumLabels[i]
		}
		for _, w := range strings.Fields(foldMessage(text)) {
			out[w] = true
		}
		return out
	}
	phrases := []string{}
	add := func(p string) {
		if p != "" && !slices.Contains(phrases, p) {
			phrases = append(phrases, p)
		}
	}
	add(foldMessage(strings.ReplaceAll(value, "_", " ")))
	if idx < 0 {
		return phrases
	}
	if idx < len(spec.EnumLabels) {
		add(foldMessage(spec.EnumLabels[idx]))
	}
	own := words(idx)
	others := map[string]bool{}
	for i := range spec.EnumValues {
		if i != idx {
			for w := range words(i) {
				others[w] = true
			}
		}
	}
	ev := strings.Fields(foldedEv)
	if len(ev) == 0 {
		return phrases
	}
	inOwn, distinctive := 0, false
	for _, w := range ev {
		if others[w] && !own[w] {
			return phrases // the span also names another choice
		}
		if own[w] {
			inOwn++
			if !others[w] {
				distinctive = true
			}
		}
	}
	if distinctive && inOwn*2 >= len(ev) {
		add(foldedEv)
	}
	return phrases
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
