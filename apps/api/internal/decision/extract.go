package decision

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

var numberedMark = regexp.MustCompile(`(\d{1,2})\s*[\.\)\:]\s*`)
var clauseSplit = regexp.MustCompile(`(?i)\s*(?:,|;|\n|\bva\b|\bvà\b)\s*`)

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
		} else if len(found) == 0 {
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
		if slot.Type != "enum" {
			continue
		}
		for _, raw := range slot.EnumValues {
			phrase := strings.ReplaceAll(Fold(raw), "_", " ")
			if phrase == "" {
				continue
			}
			if folded == phrase || containsPhrase(folded, phrase) {
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
		for _, raw := range slot.EnumValues {
			phrase := strings.ReplaceAll(Fold(raw), "_", " ")
			if folded == phrase || containsPhrase(folded, phrase) {
				return raw, true
			}
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
