package decision

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

const (
	StatusMissing   = "MISSING"
	StatusKnown     = "KNOWN"
	StatusConfirmed = "CONFIRMED"

	ActionAsk     = "ASK_MISSING_SLOTS"
	ActionDirect  = "DIRECT_ANSWER"
	ActionFinal   = "PROVIDE_FINAL_GUIDANCE"
	ActionScope   = "OUT_OF_SCOPE"
	ActionConfirm = "CONFIRM_INTENT"
)

// SlotValue is one entry in session_slot_states.slot_state.
type SlotValue struct {
	Value  any    `json:"value"`
	Status string `json:"status"`
}

func (s SlotValue) Filled() bool {
	if s.Status != StatusKnown && s.Status != StatusConfirmed {
		return false
	}
	if s.Value == nil {
		return false
	}
	if text, ok := s.Value.(string); ok && strings.TrimSpace(text) == "" {
		return false
	}
	return true
}

// Definition is the runtime slice of procedure_versions.definition.
type Definition struct {
	ProcedureCode  string
	Name           string
	IntentExamples []string
	Slots          map[string]SlotDef
	RequiredSlots  []string
	Conditional    []ConditionalSlot
	Guidance       Guidance
	Citations      []Citation
}

type SlotDef struct {
	Type       string
	Question   string
	EnumValues []string
}

type ConditionalSlot struct {
	WhenSlot string
	Equals   json.RawMessage
	Require  []string
}

type Guidance struct {
	Summary            string   `json:"summary"`
	Checklist          []string `json:"checklist"`
	WhereToSubmit      string   `json:"where_to_submit"`
	Notes              []string `json:"notes,omitempty"`
	FeeHint            string   `json:"fee_hint,omitempty"`
	ProcessingTimeHint string   `json:"processing_time_hint,omitempty"`
}

type Citation struct {
	DocID         string `json:"doc_id"`
	Title         string `json:"title"`
	SourceType    string `json:"source_type"`
	EffectiveDate string `json:"effective_date,omitempty"`
	Issuer        string `json:"issuer,omitempty"`
	PageHint      string `json:"page_hint,omitempty"`
}

type Question struct {
	Slot string `json:"slot"`
	Text string `json:"text"`
}

type definitionWire struct {
	ProcedureCode  string              `json:"procedure_code"`
	Name           string              `json:"name"`
	IntentExamples []string            `json:"intent_examples"`
	Slots          map[string]slotWire `json:"slots"`
	RequiredSlots  []string            `json:"required_slots"`
	Conditional    []conditionalWire   `json:"conditional_slots"`
	Guidance       guidanceWire        `json:"guidance"`
	Citations      []Citation          `json:"citations"`
}

type slotWire struct {
	Type       string   `json:"type"`
	Question   string   `json:"question"`
	EnumValues []string `json:"enum_values"`
}

type conditionalWire struct {
	When struct {
		Slot   string          `json:"slot"`
		Equals json.RawMessage `json:"equals"`
	} `json:"when"`
	Require []string `json:"require"`
}

type guidanceWire struct {
	Summary            string   `json:"summary"`
	Checklist          []string `json:"checklist"`
	WhereToSubmit      string   `json:"where_to_submit"`
	Notes              []string `json:"notes"`
	FeeHint            string   `json:"fee_hint"`
	ProcessingTimeHint string   `json:"processing_time_hint"`
}

// ParseDefinition decodes a procedure_versions.definition document.
func ParseDefinition(raw []byte) (Definition, error) {
	var w definitionWire
	if err := json.Unmarshal(raw, &w); err != nil {
		return Definition{}, fmt.Errorf("parse definition: %w", err)
	}
	if w.ProcedureCode == "" || w.Name == "" {
		return Definition{}, fmt.Errorf("parse definition: procedure_code and name are required")
	}
	def := Definition{
		ProcedureCode:  w.ProcedureCode,
		Name:           w.Name,
		IntentExamples: w.IntentExamples,
		Slots:          make(map[string]SlotDef, len(w.Slots)),
		RequiredSlots:  w.RequiredSlots,
		Guidance: Guidance{
			Summary:            w.Guidance.Summary,
			Checklist:          w.Guidance.Checklist,
			WhereToSubmit:      w.Guidance.WhereToSubmit,
			Notes:              w.Guidance.Notes,
			FeeHint:            w.Guidance.FeeHint,
			ProcessingTimeHint: w.Guidance.ProcessingTimeHint,
		},
		Citations: w.Citations,
	}
	for key, slot := range w.Slots {
		def.Slots[key] = SlotDef{
			Type:       slot.Type,
			Question:   slot.Question,
			EnumValues: slot.EnumValues,
		}
	}
	for _, c := range w.Conditional {
		if c.When.Slot == "" || len(c.Require) == 0 {
			continue
		}
		def.Conditional = append(def.Conditional, ConditionalSlot{
			WhenSlot: c.When.Slot,
			Equals:   c.When.Equals,
			Require:  c.Require,
		})
	}
	return def, nil
}

// EffectiveRequired is required_slots plus conditional slots whose when matches.
func (d Definition) EffectiveRequired(state map[string]SlotValue) []string {
	out := make([]string, 0, len(d.RequiredSlots))
	seen := map[string]struct{}{}
	add := func(key string) {
		if _, ok := d.Slots[key]; !ok {
			return
		}
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		out = append(out, key)
	}
	for _, key := range d.RequiredSlots {
		add(key)
	}
	for _, c := range d.Conditional {
		current, ok := state[c.WhenSlot]
		if !ok || !current.Filled() {
			continue
		}
		if !equalsValue(current.Value, c.Equals) {
			continue
		}
		for _, key := range c.Require {
			add(key)
		}
	}
	return out
}

// Missing lists effective slots that are not KNOWN or CONFIRMED, in definition order.
func (d Definition) Missing(state map[string]SlotValue) []string {
	required := d.EffectiveRequired(state)
	out := make([]string, 0)
	for _, key := range required {
		slot, ok := state[key]
		if !ok || !slot.Filled() {
			out = append(out, key)
		}
	}
	return out
}

func equalsValue(got any, want json.RawMessage) bool {
	if len(bytes.TrimSpace(want)) == 0 {
		return false
	}
	var expected any
	if err := json.Unmarshal(want, &expected); err != nil {
		return false
	}
	return valuesEqual(got, expected)
}

func valuesEqual(got, want any) bool {
	switch w := want.(type) {
	case bool:
		g, ok := got.(bool)
		return ok && g == w
	case string:
		g, ok := got.(string)
		return ok && g == w
	case float64:
		switch g := got.(type) {
		case float64:
			return g == w
		case int:
			return float64(g) == w
		case json.Number:
			f, err := g.Float64()
			return err == nil && f == w
		default:
			return false
		}
	default:
		gb, err1 := json.Marshal(got)
		wb, err2 := json.Marshal(want)
		return err1 == nil && err2 == nil && bytes.Equal(gb, wb)
	}
}
