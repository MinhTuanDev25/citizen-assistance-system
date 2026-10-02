package decision

// Policy centralizes the intent select/confirm/none confidence thresholds so
// keyword matching (MatchIntentBands) and AI-assisted intent selection never
// diverge into two different, silently-drifting notions of "confident
// enough". There is exactly one place these numbers live.
type Policy struct {
	SelectMin  float64
	ConfirmMin float64
}

// DefaultPolicy holds the original hardcoded keyword thresholds (0.82/0.55).
// MatchIntentBands below references these constants directly, so the
// pre-P2 keyword behavior is numerically unchanged. AI intent bands default
// to the same policy unless overridden via AI_INTENT_SELECT_MIN /
// AI_INTENT_CONFIRM_MIN.
var DefaultPolicy = Policy{SelectMin: 0.82, ConfirmMin: 0.55}

// Band classifies a single confidence score into select/confirm/none.
func (p Policy) Band(confidence float64) string {
	switch {
	case confidence >= p.SelectMin:
		return IntentSelect
	case confidence >= p.ConfirmMin:
		return IntentConfirm
	default:
		return IntentNone
	}
}
