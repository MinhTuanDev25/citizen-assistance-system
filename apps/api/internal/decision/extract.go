package decision

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

var numberedMark = regexp.MustCompile(`(\d{1,2})\s*[\.\)\:]\s*`)

// clauseSplit separates answers on comma, semicolon, newline, or a standalone
// "va"/"và". Go's \b is ASCII-only, so \bvà\b never matches: à is not a word
// character, and the boundary after it disappears. Whitespace on both sides
// keeps "vàng" and "advantage" intact.
var clauseSplit = regexp.MustCompile(`(?i)\s*(?:,|;|\n)\s*|\s+(?:va|và)\s+`)

var stopwords = map[string]struct{}{
	"anh": {}, "chi": {}, "da": {}, "chua": {}, "khong": {}, "co": {},
	"cua": {}, "cho": {}, "mot": {}, "cac": {}, "nao": {}, "gi": {},
	"la": {}, "duoc": {}, "hay": {}, "va": {}, "hoac": {}, "ban": {},
	"be": {}, "toi": {}, "minh": {}, "voi": {}, "trong": {}, "khi": {},
	"neu": {}, "theo": {}, "tai": {}, "den": {}, "de": {}, "thi": {},
	"nay": {}, "kia": {}, "do": {}, "the": {}, "roi": {}, "dung": {},
	"yes": {}, "ko": {}, "sai": {}, "phai": {}, "a": {}, "o": {},
}

// Extract reads values for allowed keys from a citizen message.
// It does not invent keys and does not overwrite slots the caller omits.
// The second result lists numbered answers that did not fit the slot type.
func Extract(def Definition, allowed []string, message string) (map[string]any, []string) {
	return extract(def, allowed, message, false)
}

// extractExplicit only accepts a clause that names the slot (keyword, enum, number)
// or a numbered item. Used to correct a value already saved.
func extractExplicit(def Definition, allowed []string, message string) map[string]any {
	found, _ := extract(def, allowed, message, true)
	return found
}

func extract(def Definition, allowed []string, message string, explicit bool) (map[string]any, []string) {
	allowedSet := map[string]struct{}{}
	for _, key := range allowed {
		if _, ok := def.Slots[key]; ok {
			allowedSet[key] = struct{}{}
		}
	}
	if len(allowedSet) == 0 {
		return map[string]any{}, nil
	}

	found := map[string]any{}
	var rejected []string
	for key, text := range parseNumbered(message) {
		if key < 1 || key > len(allowed) {
			continue
		}
		slotKey := allowed[key-1]
		if _, ok := allowedSet[slotKey]; !ok {
			continue
		}
		if value, ok := coerce(def.Slots[slotKey], text); ok {
			found[slotKey] = value
			delete(allowedSet, slotKey)
			continue
		}
		rejected = append(rejected, fmt.Sprintf("%d. %s", key, text))
	}

	if len(found) == 0 && !explicit {
		if zipped := zipClauses(def, allowed, message); zipped != nil {
			return zipped, nil
		}
	}

	clauses := splitClauses(message)
	for _, clause := range clauses {
		key, value, ok := matchClause(def, allowedSet, clause, !explicit)
		if !ok {
			continue
		}
		found[key] = value
		delete(allowedSet, key)
	}

	if !explicit && len(allowedSet) == 1 {
		var only string
		for key := range allowedSet {
			only = key
		}
		slot := def.Slots[only]
		if slot.Type == "string" && len(found) > 0 {
			if leftover := leftoverClauses(message, def, found); leftover != "" {
				found[only] = leftover
			}
		} else if len(found) == 0 && slot.Type != "boolean" {
			// A lone remaining boolean must match a clause with slot cues
			// (matchBooleanClause). Coercing the whole utterance would treat
			// unrelated "có"/"đã" (e.g. "đã có giấy chứng sinh") as marriage.
			if value, ok := coerce(slot, strings.TrimSpace(message)); ok {
				found[only] = value
			}
		}
	}
	return found, rejected
}

// zipClauses maps "nơi sinh, chưa, ko" onto the open slots in question order.
func zipClauses(def Definition, allowed []string, message string) map[string]any {
	clauses := splitClauses(message)
	if len(clauses) < 2 || len(clauses) != len(allowed) {
		return nil
	}
	out := make(map[string]any, len(allowed))
	for i, key := range allowed {
		slot, ok := def.Slots[key]
		if !ok {
			return nil
		}
		value, ok := coerce(slot, clauses[i])
		if !ok {
			return nil
		}
		out[key] = value
	}
	return out
}

// LoneStringAnswer fills the only open string slot from a follow-up sentence
// such as "ở bệnh viện Hùng Vương", when the citizen is already answering.
// enumOptionMatches: the folded text names enum value i, by its code
// (truc_tiep → "truc tiep") or by its label ("Trực tiếp").
func enumOptionMatches(slot SlotDef, i int, folded string) bool {
	phrases := []string{strings.ReplaceAll(Fold(slot.EnumValues[i]), "_", " ")}
	if i < len(slot.EnumLabels) {
		phrases = append(phrases, Fold(slot.EnumLabels[i]))
	}
	for _, phrase := range phrases {
		if phrase != "" && (folded == phrase || containsPhrase(folded, phrase)) {
			return true
		}
	}
	return false
}

var choiceMark = regexp.MustCompile(`^(?:(?:chon|phuong an|lua chon|y|so|dap an)\s*)?([a-z]|\d{1,2})\s*[\.\)]?$`)

// choiceIndex reads a bare choice: "b", "2", "chọn b", "phương án 2".
func choiceIndex(folded string, n int) (int, bool) {
	m := choiceMark.FindStringSubmatch(strings.TrimSpace(folded))
	if m == nil {
		return 0, false
	}
	idx := -1
	if c := m[1][0]; c >= 'a' && c <= 'z' && len(m[1]) == 1 {
		idx = int(c - 'a')
	} else if v, err := strconv.Atoi(m[1]); err == nil {
		idx = v - 1
	}
	if idx < 0 || idx >= n {
		return 0, false
	}
	return idx, true
}

// LoneChoiceAnswer: a bare choice ("b", "2") answers the only enum question
// still open. With two enum questions open it is ambiguous and not used.
func LoneChoiceAnswer(def Definition, allowed []string, message string) (string, any, bool) {
	var only string
	enums := 0
	for _, key := range allowed {
		if def.Slots[key].Type == "enum" {
			enums++
			only = key
		}
	}
	if enums != 1 {
		return "", nil, false
	}
	slot := def.Slots[only]
	folded := Fold(message)
	if i, ok := choiceIndex(folded, len(slot.EnumValues)); ok {
		return only, slot.EnumValues[i], true
	}
	if len(splitClauses(message)) != 1 || hedgedFor(slot, folded) {
		return "", nil, false
	}
	if asksChoice(slot) {
		folded = withoutSoftHedges(folded)
	}
	if i, ok := shortOptionAnswer(slot, folded); ok {
		return only, slot.EnumValues[i], true
	}
	return "", nil, false
}

// answerFiller: pronouns and particles around a short answer ("con tôi đẻ
// thường", "dạ có ạ"). Not negations or "có": they carry the polarity.
var answerFiller = map[string]bool{
	"con": true, "toi": true, "em": true, "be": true, "chau": true, "minh": true,
	"tui": true, "nha": true, "nhe": true, "a": true, "da": true, "oi": true,
	"thi": true, "la": true, "ma": true, "vang": true, "u": true,
	"luon": true, "se": true, "muon": true,
}

var answerNegation = map[string]bool{"khong": true, "chua": true, "ko": true}

// shortOptionAnswer reads a short answer to the only open choice question:
//   - every word belongs to exactly one option with the same polarity
//     ("có", "có giấy" → "Có Giấy chứng sinh", not "Không có Giấy chứng sinh…"), or
//   - it has a word only one option uses and none of another's
//     ("đẻ thường" → "Trẻ sinh thông thường").
//
// Hedged answers never reach here.
func shortOptionAnswer(slot SlotDef, folded string) (int, bool) {
	var words []string
	negated := false
	for _, w := range strings.Fields(folded) {
		if answerFiller[w] {
			continue
		}
		if answerNegation[w] {
			negated = true
		}
		words = append(words, w)
	}
	if len(words) == 0 || len(words) > 6 {
		return 0, false
	}
	n := len(slot.EnumValues)
	own := make([]map[string]bool, n)
	neg := make([]bool, n)
	for i, v := range slot.EnumValues {
		own[i] = map[string]bool{}
		text := strings.ReplaceAll(v, "_", " ")
		if i < len(slot.EnumLabels) {
			text += " " + slot.EnumLabels[i]
		}
		for _, w := range strings.Fields(Fold(text)) {
			own[i][w] = true
		}
		if i < len(slot.EnumLabels) {
			for _, w := range strings.Fields(Fold(slot.EnumLabels[i])) {
				if answerNegation[w] {
					neg[i] = true
				}
			}
		}
	}
	match := -1
	for i := range own {
		if neg[i] != negated {
			continue
		}
		all := true
		for _, w := range words {
			if !own[i][w] {
				all = false
				break
			}
		}
		if all {
			if match >= 0 {
				match = -2
				break
			}
			match = i
		}
	}
	if match >= 0 {
		return match, true
	}
	if negated {
		return 0, false
	}
	// A word that only one option uses names it, when no word names another.
	hit := -1
	for _, w := range words {
		users := []int{}
		for i := range own {
			if own[i][w] {
				users = append(users, i)
			}
		}
		if len(users) != 1 {
			continue
		}
		if hit >= 0 && hit != users[0] {
			return 0, false
		}
		hit = users[0]
	}
	if hit < 0 || neg[hit] {
		return 0, false
	}
	return hit, true
}

func LoneStringAnswer(def Definition, allowed []string, message string) (string, any, bool) {
	if len(splitClauses(message)) != 1 {
		return "", nil, false
	}
	folded := Fold(message)
	if folded == "" || isChatter(folded) {
		return "", nil, false
	}
	if _, ok := polarity(folded); ok && len(tokens(folded)) <= 4 {
		return "", nil, false
	}
	var only string
	stringsN := 0
	for _, key := range allowed {
		if def.Slots[key].Type == "string" {
			stringsN++
			only = key
		}
	}
	if stringsN != 1 {
		return "", nil, false
	}
	value, ok := coerce(def.Slots[only], strings.TrimSpace(message))
	if !ok {
		return "", nil, false
	}
	return only, value, true
}

func isChatter(folded string) bool {
	switch folded {
	case "xin chao", "chao", "hello", "hi", "cam on", "ok", "da", "vang", "roi":
		return true
	default:
		return false
	}
}

func parseNumbered(message string) map[int]string {
	marks := numberedMark.FindAllStringSubmatchIndex(message, -1)
	if len(marks) == 0 {
		return nil
	}
	out := map[int]string{}
	sawOne := false
	for i, mark := range marks {
		n, err := strconv.Atoi(message[mark[2]:mark[3]])
		if err != nil {
			continue
		}
		if n == 1 {
			sawOne = true
		}
		end := len(message)
		if i+1 < len(marks) {
			end = marks[i+1][0]
		}
		text := strings.TrimSpace(message[mark[1]:end])
		text = strings.Trim(text, " \t\r\n,;.")
		if text == "" {
			continue
		}
		out[n] = text
	}
	if !sawOne {
		return nil
	}
	return out
}

func splitClauses(message string) []string {
	parts := clauseSplit.Split(message, -1)
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func matchClause(def Definition, allowed map[string]struct{}, clause string, allowBare bool) (string, any, bool) {
	folded := Fold(clause)
	if folded == "" {
		return "", nil, false
	}

	if key, value, ok := matchEnumClause(def, allowed, folded); ok {
		return key, value, true
	}
	if key, value, ok := matchNumberClause(def, allowed, clause, folded); ok {
		return key, value, true
	}
	if key, value, ok := matchBooleanClause(def, allowed, folded, allowBare); ok {
		return key, value, true
	}
	return "", nil, false
}

func matchEnumClause(def Definition, allowed map[string]struct{}, folded string) (string, any, bool) {
	var hitKey string
	var hitValue string
	hits := 0
	for key := range allowed {
		slot := def.Slots[key]
		if slot.Type != "enum" || hedgedFor(slot, folded) {
			continue
		}
		for i, raw := range slot.EnumValues {
			if enumOptionMatches(slot, i, folded) {
				hits++
				hitKey = key
				hitValue = raw
			}
		}
	}
	if hits == 1 {
		return hitKey, hitValue, true
	}
	return "", nil, false
}

func matchNumberClause(def Definition, allowed map[string]struct{}, clause, folded string) (string, any, bool) {
	if _, err := strconv.ParseFloat(strings.ReplaceAll(folded, " ", ""), 64); err != nil && !mostlyNumber(folded) {
		return "", nil, false
	}
	var only string
	count := 0
	for key := range allowed {
		if def.Slots[key].Type == "number" {
			count++
			only = key
		}
	}
	if count != 1 {
		return "", nil, false
	}
	value, ok := coerce(def.Slots[only], clause)
	if !ok {
		return "", nil, false
	}
	return only, value, true
}

func mostlyNumber(folded string) bool {
	digits := 0
	for _, r := range folded {
		if r >= '0' && r <= '9' {
			digits++
		}
	}
	return digits > 0 && digits >= len(strings.ReplaceAll(folded, " ", ""))/2
}

func matchBooleanClause(def Definition, allowed map[string]struct{}, folded string, allowBare bool) (string, any, bool) {
	if isUncertainFolded(folded) {
		return "", nil, false
	}
	pol, ok := polarity(folded)
	if !ok {
		return "", nil, false
	}

	type hit struct {
		key   string
		score int
	}
	var hits []hit
	for key := range allowed {
		slot := def.Slots[key]
		if slot.Type != "boolean" {
			continue
		}
		score := keywordScore(slot.Question, folded)
		if score >= 2 {
			hits = append(hits, hit{key: key, score: score})
		}
	}
	if len(hits) == 0 {
		if allowBare && len(folded) <= 12 && boolCount(def, allowed) == 1 {
			var only string
			for key := range allowed {
				if def.Slots[key].Type == "boolean" {
					only = key
				}
			}
			return only, pol, true
		}
		return "", nil, false
	}
	best := hits[0]
	tie := false
	for _, h := range hits[1:] {
		if h.score > best.score {
			best = h
			tie = false
		} else if h.score == best.score {
			tie = true
		}
	}
	if tie {
		return "", nil, false
	}
	return best.key, pol, true
}

func boolCount(def Definition, allowed map[string]struct{}) int {
	n := 0
	for key := range allowed {
		if def.Slots[key].Type == "boolean" {
			n++
		}
	}
	return n
}

func keywordScore(question, foldedClause string) int {
	score := 0
	for _, token := range tokens(Fold(question)) {
		if len(token) < 3 {
			continue
		}
		if _, skip := stopwords[token]; skip {
			continue
		}
		if containsPhrase(foldedClause, token) {
			score++
		}
	}
	return score
}

var (
	uncertainPhrases = []string{
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
		"khong muon",
	}
	// Soften a choice without leaving it open ("chắc là tôi muốn trực tiếp").
	softHedges = map[string]bool{"hinh nhu": true, "co le": true, "chac la": true, "co the se": true, "co the": true}
	// A question with these words asks what the citizen wants or chooses.
	choiceQuestionPhrases = []string{"muon", "lua chon", "chon", "du dinh", "du kien"}
)

func isUncertainFolded(folded string) bool {
	for _, p := range uncertainPhrases {
		if containsPhrase(folded, p) {
			return true
		}
	}
	return false
}

// isUndecidedFolded: uncertain even for a choice ("không biết", "chưa chắc",
// "không muốn"), not just softened.
func isUndecidedFolded(folded string) bool {
	for _, p := range uncertainPhrases {
		if !softHedges[p] && containsPhrase(folded, p) {
			return true
		}
	}
	return false
}

func asksChoice(slot SlotDef) bool {
	folded := Fold(slot.Question)
	for _, p := range choiceQuestionPhrases {
		if containsPhrase(folded, p) {
			return true
		}
	}
	return false
}

// hedgedFor: the clause is too unsure to answer slot. A choice belongs to the
// citizen, so a soft hedge still picks it; a fact ("chắc là có giấy") does not.
func hedgedFor(slot SlotDef, folded string) bool {
	if slot.Type == "enum" && asksChoice(slot) {
		return isUndecidedFolded(folded)
	}
	return isUncertainFolded(folded)
}

// withoutSoftHedges drops softening words before matching a short answer.
func withoutSoftHedges(folded string) string {
	padded := " " + folded + " "
	for p := range softHedges {
		padded = strings.ReplaceAll(padded, " "+p+" ", " ")
	}
	return strings.TrimSpace(padded)
}

func polarity(folded string) (bool, bool) {
	switch {
	case containsPhrase(folded, "khong co") || containsPhrase(folded, "chua co") || containsPhrase(folded, "chua"):
		return false, true
	case containsPhrase(folded, "khong") || containsPhrase(folded, "ko") || containsPhrase(folded, "sai"):
		return false, true
	case containsPhrase(folded, "co") || containsPhrase(folded, "roi") || containsPhrase(folded, "dung") || containsPhrase(folded, "da") || containsPhrase(folded, "phai"):
		return true, true
	default:
		return false, false
	}
}

func coerce(slot SlotDef, text string) (any, bool) {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, false
	}
	switch slot.Type {
	case "boolean":
		return polarity(Fold(text))
	case "number":
		cleaned := strings.ReplaceAll(Fold(text), " ", "")
		re := regexp.MustCompile(`\d+(?:\.\d+)?`)
		num := re.FindString(cleaned)
		if num == "" {
			return nil, false
		}
		f, err := strconv.ParseFloat(num, 64)
		if err != nil {
			return nil, false
		}
		return f, true
	case "enum":
		folded := Fold(text)
		for i, raw := range slot.EnumValues {
			if enumOptionMatches(slot, i, folded) {
				return raw, true
			}
		}
		if i, ok := choiceIndex(folded, len(slot.EnumValues)); ok {
			return slot.EnumValues[i], true
		}
		return nil, false
	default:
		if len([]rune(text)) < 2 {
			return nil, false
		}
		return text, true
	}
}

func leftoverClauses(message string, def Definition, found map[string]any) string {
	var parts []string
	for _, clause := range splitClauses(message) {
		folded := Fold(clause)
		used := false
		for key := range found {
			slot := def.Slots[key]
			if slot.Type == "boolean" && keywordScore(slot.Question, folded) >= 2 {
				used = true
				break
			}
			if slot.Type == "enum" {
				if value, ok := found[key].(string); ok {
					phrase := strings.ReplaceAll(Fold(value), "_", " ")
					if containsPhrase(folded, phrase) {
						used = true
						break
					}
				}
			}
		}
		if !used {
			parts = append(parts, strings.TrimSpace(clause))
		}
	}
	return strings.TrimSpace(strings.Join(parts, ", "))
}
