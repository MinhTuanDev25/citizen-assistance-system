package aiextract

import (
	"fmt"
	"math"
	"regexp"
	"strings"
)

const (
	maxProviderLen     = 32
	maxModelLen        = 80
	maxAbstainRunes    = 300
	maxEvidenceRunes   = 300
	maxStringSlotRunes = 500
)

// modelNamePattern rejects free text, phone numbers, and whitespace so a
// model-controlled string cannot carry a citizen message into metadata.
var modelNamePattern = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9._:/-]{0,79}$`)

// SafeProviderModel accepts only the configured provider names and a short
// token-shaped model id. The error text never echoes the rejected values.
func SafeProviderModel(provider, model string) (string, string, error) {
	provider = strings.TrimSpace(provider)
	model = strings.TrimSpace(model)
	switch provider {
	case "mock", "openai", "gemini":
	default:
		return "", "", fmt.Errorf("aiextract: invalid provider")
	}
	if len(provider) > maxProviderLen || len(model) > maxModelLen || !modelNamePattern.MatchString(model) {
		return "", "", fmt.Errorf("aiextract: invalid model")
	}
	return provider, model, nil
}

// CurrentSlot is the minimal state Validate needs to decide whether a
// "correct" operation is legal (it never is against an empty/missing slot).
type CurrentSlot struct {
	Filled bool
}

// ValidatedSlot is one Go-validated, type/enum-checked slot proposal.
type ValidatedSlot struct {
	Value      any
	Operation  string // "set" | "correct"
	Confidence float64
}

// ValidatedProposal is the only shape the chat/decision layer is ever
// allowed to see from an AI response — every field has already passed every
// fail-closed rule below. IntentProcedureCode and SlotsProcedureCode are
// tracked separately because the contract allows a response to name an
// intent switch candidate while still supplying slots for the *currently
// pinned* procedure (or vice versa); the caller decides what to do with each
// independently. Provider and Model are the allowlisted observability values;
// evidence, abstain text, and raw payloads are not copied out.
type ValidatedProposal struct {
	IntentProcedureCode string
	IntentConfidence    float64
	Alternatives        []Alternative

	SlotsProcedureCode string
	Slots              map[string]ValidatedSlot

	Provider string
	Model    string
}

func isFiniteUnitInterval(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0) && v >= 0 && v <= 1
}

// Validate re-derives a trustworthy proposal from a raw *Response.
//
// Any violation discards the ENTIRE response — there is no partial-apply.
// candidates is the exact candidate list Go sent this turn (so an unknown
// procedure_code/slot key can never sneak through). pinnedProcedureCode is
// "" when no procedure is pinned yet. allowedSlotKeys is enforced only when
// the slots target the already-pinned procedure (a fresh intent selection
// has no prior restriction — any slot declared on that candidate's own
// definition is acceptable). currentState carries Filled() per slot key so
// "correct" can never blindly overwrite an empty slot.
func Validate(
	resp *Response,
	candidates []Candidate,
	pinnedProcedureCode string,
	allowedSlotKeys []string,
	slotConfidenceMin float64,
	currentState map[string]CurrentSlot,
) (*ValidatedProposal, error) {
	if resp == nil {
		return nil, fmt.Errorf("aiextract: nil response")
	}
	if resp.SchemaVersion != SchemaVersion {
		return nil, fmt.Errorf("aiextract: unexpected schema_version %q", resp.SchemaVersion)
	}
	provider, model, err := SafeProviderModel(resp.Provider, resp.Model)
	if err != nil {
		return nil, err
	}
	if resp.AbstainReason != nil && len([]rune(*resp.AbstainReason)) > maxAbstainRunes {
		return nil, fmt.Errorf("aiextract: abstain_reason exceeds limit")
	}

	byCode := map[string]Candidate{}
	for _, c := range candidates {
		if _, dup := byCode[c.ProcedureCode]; dup {
			return nil, fmt.Errorf("aiextract: duplicate candidate %q", c.ProcedureCode)
		}
		byCode[c.ProcedureCode] = c
	}

	out := &ValidatedProposal{Slots: map[string]ValidatedSlot{}, Provider: provider, Model: model}

	if resp.Intent != nil {
		alts, err := validateAlternatives(resp.Intent.Alternatives, byCode)
		if err != nil {
			return nil, err
		}
		code := ""
		if resp.Intent.ProcedureCode != nil {
			code = strings.TrimSpace(*resp.Intent.ProcedureCode)
		}
		if code == "" {
			// Null or blank intent must not claim confidence. Alternatives are
			// checked above and then dropped: a null pick does not select or
			// confirm anything.
			if resp.Intent.Confidence != 0 {
				return nil, fmt.Errorf("aiextract: null intent confidence must be 0")
			}
		} else {
			if _, ok := byCode[code]; !ok {
				return nil, fmt.Errorf("aiextract: unknown intent procedure_code %q", code)
			}
			if !isFiniteUnitInterval(resp.Intent.Confidence) {
				return nil, fmt.Errorf("aiextract: intent confidence out of range")
			}
			out.IntentProcedureCode = code
			out.IntentConfidence = resp.Intent.Confidence
			out.Alternatives = alts
		}
	}

	if resp.SlotsForProcedureCode != nil {
		sfp := strings.TrimSpace(*resp.SlotsForProcedureCode)
		if sfp != "" {
			if sfp != out.IntentProcedureCode && sfp != pinnedProcedureCode {
				return nil, fmt.Errorf(
					"aiextract: slots_for_procedure_code %q matches neither the selected intent nor the pinned procedure", sfp,
				)
			}
			out.SlotsProcedureCode = sfp
		}
	}
	if len(resp.Slots) > 0 && out.SlotsProcedureCode == "" {
		return nil, fmt.Errorf("aiextract: slots present without a valid slots_for_procedure_code")
	}
	if len(resp.Slots) == 0 {
		return out, nil
	}

	targetCandidate, ok := byCode[out.SlotsProcedureCode]
	if !ok {
		return nil, fmt.Errorf("aiextract: slots target unknown procedure %q", out.SlotsProcedureCode)
	}

	// allowed_slot_keys only restricts further when the AI is filling the
	// procedure that is *already pinned* — a fresh selection has no prior
	// restriction context, so any slot declared on that candidate is fine.
	enforceAllowed := out.SlotsProcedureCode != "" && out.SlotsProcedureCode == pinnedProcedureCode
	allowedSet := map[string]struct{}{}
	for _, k := range allowedSlotKeys {
		allowedSet[k] = struct{}{}
	}

	seenKeys := map[string]struct{}{}
	for _, s := range resp.Slots {
		if _, dup := seenKeys[s.Key]; dup {
			return nil, fmt.Errorf("aiextract: duplicate slot key %q in response", s.Key)
		}
		seenKeys[s.Key] = struct{}{}

		spec, ok := targetCandidate.Slots[s.Key]
		if !ok {
			return nil, fmt.Errorf("aiextract: unknown slot key %q for procedure %q", s.Key, out.SlotsProcedureCode)
		}
		if enforceAllowed {
			if _, ok := allowedSet[s.Key]; !ok {
				return nil, fmt.Errorf("aiextract: slot key %q is not in allowed_slot_keys", s.Key)
			}
		}
		if len([]rune(s.Evidence)) > maxEvidenceRunes {
			return nil, fmt.Errorf("aiextract: slot %q evidence exceeds limit", s.Key)
		}
		if s.Operation != "set" && s.Operation != "correct" {
			return nil, fmt.Errorf("aiextract: invalid operation %q for slot %q", s.Operation, s.Key)
		}
		if !isFiniteUnitInterval(s.Confidence) {
			return nil, fmt.Errorf("aiextract: slot %q confidence out of range", s.Key)
		}
		if s.Operation == "correct" && !currentState[s.Key].Filled {
			return nil, fmt.Errorf("aiextract: slot %q operation=correct but slot is not currently filled", s.Key)
		}

		value, err := coerceTyped(spec, s.Value)
		if err != nil {
			return nil, fmt.Errorf("aiextract: slot %q: %w", s.Key, err)
		}

		if s.Confidence < slotConfidenceMin {
			// Below threshold: drop only this slot, not the whole response —
			// this is a policy filter, not a contract violation.
			continue
		}
		out.Slots[s.Key] = ValidatedSlot{Value: value, Operation: s.Operation, Confidence: s.Confidence}
	}
	return out, nil
}

func validateAlternatives(alts []Alternative, byCode map[string]Candidate) ([]Alternative, error) {
	seen := map[string]struct{}{}
	out := make([]Alternative, 0, len(alts))
	for _, alt := range alts {
		altCode := strings.TrimSpace(alt.ProcedureCode)
		if _, dup := seen[altCode]; dup {
			return nil, fmt.Errorf("aiextract: duplicate alternative %q", altCode)
		}
		seen[altCode] = struct{}{}
		if _, ok := byCode[altCode]; !ok {
			return nil, fmt.Errorf("aiextract: unknown alternative procedure_code %q", altCode)
		}
		if !isFiniteUnitInterval(alt.Confidence) {
			return nil, fmt.Errorf("aiextract: alternative confidence out of range")
		}
		out = append(out, Alternative{ProcedureCode: altCode, Confidence: alt.Confidence})
	}
	return out, nil
}

func coerceTyped(spec SlotSpec, raw any) (any, error) {
	switch spec.Type {
	case "boolean":
		b, ok := raw.(bool)
		if !ok {
			return nil, fmt.Errorf("expected boolean value")
		}
		return b, nil
	case "number":
		f, ok := raw.(float64)
		if !ok {
			return nil, fmt.Errorf("expected number value")
		}
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil, fmt.Errorf("number value must be finite")
		}
		return f, nil
	case "enum":
		s, ok := raw.(string)
		if !ok {
			return nil, fmt.Errorf("expected string value for enum")
		}
		for _, allowed := range spec.EnumValues {
			if s == allowed {
				return s, nil
			}
		}
		return nil, fmt.Errorf("value %q does not match any enum_values", s)
	case "string":
		s, ok := raw.(string)
		if !ok {
			return nil, fmt.Errorf("expected string value")
		}
		trimmed := strings.TrimSpace(s)
		if trimmed == "" {
			return nil, fmt.Errorf("empty string value")
		}
		if len([]rune(trimmed)) > maxStringSlotRunes {
			return nil, fmt.Errorf("string value exceeds %d runes", maxStringSlotRunes)
		}
		return trimmed, nil
	default:
		return nil, fmt.Errorf("unknown slot type %q", spec.Type)
	}
}
