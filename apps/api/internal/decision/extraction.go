package decision

import "sort"

// ExtractedSlot is one Go-validated (aiextract.Validate has already run)
// slot value the AI proposed. decision stays LLM-agnostic: it only ever sees
// a plain key -> value/operation pair, never a raw provider response.
type ExtractedSlot struct {
	Value     any
	Operation string // "set" | "correct"
}

// ExtractionProposal is what DecideWithExtraction seeds into prior state
// before delegating to the unchanged Decide pipeline.
type ExtractionProposal struct {
	Slots map[string]ExtractedSlot
}

// AppliedSlots records which AI-proposed keys were actually merged in vs.
// rejected, for non-PII observability metadata (never the citizen message).
type AppliedSlots struct {
	Accepted []string
	Rejected []string
}

// DecideWithExtraction merges a validated AI slot proposal into TurnInput's
// prior state before running the deterministic Decide pipeline unchanged.
//
// This is the ONLY way the AI can influence the decision: it may pre-fill or
// correct slot values. It can never choose the resulting Action, never
// invents a slot outside the definition, never touches Guidance/Checklist/
// Citations (those come from Decide exactly as before P2), and a "set" never
// overwrites an already-filled value (mirrors the keyword engine's own
// invariant — corrections must go through "correct" against a filled slot).
func DecideWithExtraction(in TurnInput, extraction *ExtractionProposal) (TurnOutput, AppliedSlots) {
	if extraction == nil || len(extraction.Slots) == 0 {
		return Decide(in), AppliedSlots{}
	}

	seeded := cloneState(in.Prior)
	var applied AppliedSlots

	keys := make([]string, 0, len(extraction.Slots))
	for k := range extraction.Slots {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, key := range keys {
		es := extraction.Slots[key]
		if _, known := in.Definition.Slots[key]; !known {
			applied.Rejected = append(applied.Rejected, key)
			continue
		}
		current := seeded[key]
		switch es.Operation {
		case "set":
			if current.Filled() {
				applied.Rejected = append(applied.Rejected, key)
				continue
			}
			seeded[key] = SlotValue{Value: es.Value, Status: StatusKnown}
			applied.Accepted = append(applied.Accepted, key)
		case "correct":
			if !current.Filled() || valuesEqual(current.Value, es.Value) {
				applied.Rejected = append(applied.Rejected, key)
				continue
			}
			seeded[key] = SlotValue{Value: es.Value, Status: StatusKnown}
			applied.Accepted = append(applied.Accepted, key)
		default:
			applied.Rejected = append(applied.Rejected, key)
		}
	}

	out := Decide(TurnInput{
		Definition:    in.Definition,
		Prior:         seeded,
		HadPriorState: in.HadPriorState,
		Message:       in.Message,
	})
	return out, applied
}
