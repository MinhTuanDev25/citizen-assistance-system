package decision

import "strings"

// IntentCandidate is one ACTIVE procedure the matcher may select.
type IntentCandidate struct {
	ProcedureCode string
	Name          string
	Examples      []string
}

// RankedIntent is a scored candidate for confirmation UI.
type RankedIntent struct {
	ProcedureCode string  `json:"procedure_code"`
	Name          string  `json:"name"`
	Score         float64 `json:"score"`
}

const (
	IntentSelect  = "select"
	IntentConfirm = "confirm"
	IntentNone    = "none"
)

// IntentMatch is the three-band matcher result.
type IntentMatch struct {
	Band       string // select | confirm | none
	Selected   IntentCandidate
	Candidates []RankedIntent
}

// MatchIntentBands picks select / confirm / out-of-scope from folded scores.
func MatchIntentBands(message string, candidates []IntentCandidate) IntentMatch {
	folded := Fold(message)
	if folded == "" || len(candidates) == 0 {
		return IntentMatch{Band: IntentNone}
	}

	type scored struct {
		item  IntentCandidate
		score float64
	}
	ranked := make([]scored, 0, len(candidates))
	for _, c := range candidates {
		ranked = append(ranked, scored{item: c, score: scoreCandidate(folded, c)})
	}
	// sort desc
	for i := 0; i < len(ranked); i++ {
		for j := i + 1; j < len(ranked); j++ {
			if ranked[j].score > ranked[i].score {
				ranked[i], ranked[j] = ranked[j], ranked[i]
			}
		}
	}

	best := ranked[0]
	second := 0.0
	if len(ranked) > 1 {
		second = ranked[1].score
	}

	toRanked := func(limit int) []RankedIntent {
		out := make([]RankedIntent, 0, limit)
		for i, r := range ranked {
			if i >= limit || r.score < 0.45 {
				break
			}
			out = append(out, RankedIntent{
				ProcedureCode: r.item.ProcedureCode,
				Name:          r.item.Name,
				Score:         r.score,
			})
		}
		return out
	}

	// High confidence: clear winner.
	if best.score >= DefaultPolicy.SelectMin && best.score-second >= 0.05 {
		return IntentMatch{Band: IntentSelect, Selected: best.item, Candidates: toRanked(1)}
	}
	// Mid / near-tie: ask confirmation.
	if best.score >= DefaultPolicy.ConfirmMin {
		cands := toRanked(3)
		if len(cands) == 0 {
			return IntentMatch{Band: IntentNone}
		}
		return IntentMatch{Band: IntentConfirm, Candidates: cands}
	}
	return IntentMatch{Band: IntentNone}
}

// MatchIntent keeps the binary helper for older tests (select only).
func MatchIntent(message string, candidates []IntentCandidate) (IntentCandidate, bool) {
	m := MatchIntentBands(message, candidates)
	if m.Band != IntentSelect {
		return IntentCandidate{}, false
	}
	return m.Selected, true
}

func scoreCandidate(foldedMessage string, c IntentCandidate) float64 {
	best := scoreText(foldedMessage, c.Name)
	for _, example := range c.Examples {
		if s := scoreText(foldedMessage, example); s > best {
			best = s
		}
	}
	return best
}

func scoreText(foldedMessage, raw string) float64 {
	folded := Fold(raw)
	if folded == "" || foldedMessage == "" {
		return 0
	}
	if foldedMessage == folded {
		return 1
	}
	if strings.Contains(foldedMessage, folded) && len([]rune(folded)) >= 6 {
		return 0.95
	}
	if strings.Contains(folded, foldedMessage) && len([]rune(foldedMessage)) >= 6 {
		return 0.9
	}
	phrase := distinctivePhraseScore(foldedMessage, folded)
	fuzzy := fuzzyTokenScore(foldedMessage, folded)
	et := tokens(folded)
	if len(et) == 0 {
		return maxf(phrase, fuzzy)
	}
	mt := map[string]struct{}{}
	for _, t := range tokens(foldedMessage) {
		mt[t] = struct{}{}
	}
	shared := 0
	for _, t := range et {
		if _, ok := mt[t]; ok {
			shared++
			continue
		}
		// typo-tolerant token hit
		for mtTok := range mt {
			if tokenSimilarity(t, mtTok) >= 0.75 {
				shared++
				break
			}
		}
	}
	overlap := 0.0
	if shared > 0 {
		overlap = float64(shared) / float64(len(et))
	}
	return maxf(phrase, maxf(overlap, fuzzy))
}

func fuzzyTokenScore(foldedMessage, foldedExample string) float64 {
	mt := tokens(foldedMessage)
	et := tokens(foldedExample)
	if len(mt) == 0 || len(et) == 0 {
		return 0
	}
	// distinctive example tokens (skip weak) vs message tokens
	hits := 0
	need := 0
	for _, t := range et {
		if _, weak := weakToken[t]; weak || len(t) < 3 {
			continue
		}
		need++
		best := 0.0
		for _, m := range mt {
			if s := tokenSimilarity(t, m); s > best {
				best = s
			}
		}
		if best >= 0.75 {
			hits++
		}
	}
	if need == 0 {
		return 0
	}
	ratio := float64(hits) / float64(need)
	if ratio < 0.5 {
		return 0
	}
	return 0.7 + 0.2*ratio
}

func maxf(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

var weakToken = map[string]struct{}{
	"toi": {}, "muon": {}, "lam": {}, "hoi": {}, "ve": {}, "thu": {}, "tuc": {},
	"anh": {}, "chi": {}, "mot": {}, "cai": {}, "cho": {}, "cua": {}, "nay": {},
	"ban": {}, "minh": {},
}

func distinctivePhraseScore(foldedMessage, foldedExample string) float64 {
	et := tokens(foldedExample)
	best := 0.0
	for n := 2; n <= 4 && n <= len(et); n++ {
		for i := 0; i+n <= len(et); i++ {
			if allWeak(et[i : i+n]) {
				continue
			}
			phrase := strings.Join(et[i:i+n], " ")
			if len([]rune(phrase)) < 6 {
				continue
			}
			if strings.Contains(foldedMessage, phrase) {
				score := 0.88 + float64(n)*0.02
				if score > best {
					best = score
				}
				continue
			}
			// fuzzy phrase: allow one token typo
			pt := et[i : i+n]
			mt := tokens(foldedMessage)
			if fuzzyPhraseHit(mt, pt) {
				score := 0.8 + float64(n)*0.02
				if score > best {
					best = score
				}
			}
		}
	}
	return best
}

func fuzzyPhraseHit(messageTokens, phraseTokens []string) bool {
	if len(phraseTokens) == 0 || len(messageTokens) < len(phraseTokens) {
		return false
	}
	for i := 0; i+len(phraseTokens) <= len(messageTokens); i++ {
		ok := true
		for j := range phraseTokens {
			if tokenSimilarity(messageTokens[i+j], phraseTokens[j]) < 0.75 {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

func allWeak(tokens []string) bool {
	for _, token := range tokens {
		if _, ok := weakToken[token]; !ok {
			return false
		}
	}
	return true
}
