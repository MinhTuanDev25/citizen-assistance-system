package decision

import (
	"fmt"
	"strings"
)

// TurnInput is one citizen utterance against an already selected definition.
type TurnInput struct {
	Definition    Definition
	Prior         map[string]SlotValue
	HadPriorState bool
	Message       string
}

// TurnOutput is the decision contract for one turn.
type TurnOutput struct {
	Action       string
	SlotState    map[string]SlotValue
	MissingSlots []string
	AskNow       []string
	Questions    []Question
	ReplyText    string
	Guidance     *Guidance
	Citations    []Citation
	FilledSlots  map[string]any
}

// Decide applies extract → merge → missing slots → one of the three actions.
// The caller persists the result. LLM paraphrase is intentionally not used.
func Decide(in TurnInput) TurnOutput {
	state := cloneState(in.Prior)
	missingBefore := in.Definition.Missing(state)
	extracted, rejected := Extract(in.Definition, missingBefore, in.Message)
	corrections := extractExplicit(in.Definition, filledChoices(in.Definition, state), in.Message)
	if in.HadPriorState && len(extracted) == 0 && len(rejected) == 0 && len(corrections) == 0 {
		if key, value, ok := LoneStringAnswer(in.Definition, missingBefore, in.Message); ok {
			extracted[key] = value
		}
	}
	for key, value := range extracted {
		current := state[key]
		if current.Filled() {
			continue
		}
		state[key] = SlotValue{Value: value, Status: StatusKnown}
	}
	var updated []string
	for key, value := range corrections {
		current := state[key]
		if !current.Filled() || valuesEqual(current.Value, value) {
			continue
		}
		state[key] = SlotValue{Value: value, Status: StatusKnown}
		updated = append(updated, changeLine(in.Definition, key, value))
	}

	// A conditional slot can become required only after this message is merged.
	// Read it from the same utterance when the citizen already answered it.
	missingAfter := in.Definition.Missing(state)
	var extra []string
	for _, key := range missingAfter {
		if !containsStr(missingBefore, key) {
			extra = append(extra, key)
		}
	}
	if len(extra) > 0 {
		extraFound, _ := Extract(in.Definition, extra, in.Message)
		for key, value := range extraFound {
			current := state[key]
			if current.Filled() {
				continue
			}
			state[key] = SlotValue{Value: value, Status: StatusKnown}
		}
		missingAfter = in.Definition.Missing(state)
	}

	for _, key := range in.Definition.EffectiveRequired(state) {
		if _, ok := state[key]; !ok {
			state[key] = SlotValue{Value: nil, Status: StatusMissing}
		}
	}

	out := TurnOutput{
		SlotState:    state,
		MissingSlots: missingAfter,
		Citations:    in.Definition.Citations,
	}
	if len(missingAfter) > 0 {
		out.Action = ActionAsk
		out.AskNow = append([]string{}, missingAfter...)
		out.Questions = questionsFor(in.Definition, missingAfter)
		out.ReplyText = askReply(in.Definition.Name, out.Questions)
		out.ReplyText = prefaceReply(updated, rejected, out.ReplyText)
		out.Citations = nil
		return out
	}

	out.FilledSlots = filledSlots(state, in.Definition.EffectiveRequired(state))
	out.Guidance = &in.Definition.Guidance
	out.ReplyText = prefaceReply(updated, rejected, guidanceReply(in.Definition.Guidance, in.Definition.Citations))
	if len(in.Definition.RequiredSlots) == 0 || !in.HadPriorState {
		out.Action = ActionDirect
		return out
	}
	out.Action = ActionFinal
	return out
}

// HasSlotCorrection reports whether the message changes a boolean or enum
// that is already saved. A finished procedure can still be corrected.
func HasSlotCorrection(def Definition, state map[string]SlotValue, message string) bool {
	for key, value := range extractExplicit(def, filledChoices(def, state), message) {
		current := state[key]
		if current.Filled() && !valuesEqual(current.Value, value) {
			return true
		}
	}
	return false
}

func filledChoices(def Definition, state map[string]SlotValue) []string {
	var keys []string
	for key, slot := range def.Slots {
		if slot.Type != "boolean" && slot.Type != "enum" {
			continue
		}
		if state[key].Filled() {
			keys = append(keys, key)
		}
	}
	return keys
}

func changeLine(def Definition, key string, value any) string {
	question := strings.TrimSpace(def.Slots[key].Question)
	switch v := value.(type) {
	case bool:
		if v {
			return question + " — Có."
		}
		if strings.Contains(Fold(question), "chua") && !strings.Contains(Fold(question), "khong") {
			return question + " — Chưa."
		}
		return question + " — Không."
	default:
		return question + " — " + fmt.Sprint(v) + "."
	}
}

func prefaceReply(updated, rejected []string, body string) string {
	var b strings.Builder
	if len(updated) > 0 {
		b.WriteString("Mình cập nhật:\n")
		for _, line := range updated {
			b.WriteString("- ")
			b.WriteString(line)
			b.WriteByte('\n')
		}
		b.WriteByte('\n')
	}
	if len(rejected) > 0 {
		b.WriteString("Mình chưa hiểu: ")
		b.WriteString(strings.Join(rejected, "; "))
		b.WriteString(".\n\n")
	}
	b.WriteString(body)
	return strings.TrimRight(b.String(), "\n")
}

func questionsFor(def Definition, keys []string) []Question {
	out := make([]Question, 0, len(keys))
	for _, key := range keys {
		slot, ok := def.Slots[key]
		if !ok || strings.TrimSpace(slot.Question) == "" {
			continue
		}
		out = append(out, Question{Slot: key, Text: slot.Question})
	}
	return out
}

func askReply(name string, questions []Question) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Để hướng dẫn thủ tục %s, anh/chị cho mình biết:\n", name)
	for i, q := range questions {
		fmt.Fprintf(&b, "%d) %s\n", i+1, q.Text)
	}
	return strings.TrimRight(b.String(), "\n")
}

func guidanceReply(g Guidance, citations []Citation) string {
	var b strings.Builder
	if g.Summary != "" {
		b.WriteString(g.Summary)
		b.WriteString("\n\n")
	}
	if len(g.Checklist) > 0 {
		b.WriteString("Hồ sơ cần chuẩn bị:\n")
		for i, item := range g.Checklist {
			fmt.Fprintf(&b, "%d) %s\n", i+1, item)
		}
		b.WriteByte('\n')
	}
	if g.WhereToSubmit != "" {
		fmt.Fprintf(&b, "Nơi nộp: %s\n", g.WhereToSubmit)
	}
	if g.FeeHint != "" {
		fmt.Fprintf(&b, "Lệ phí: %s\n", g.FeeHint)
	}
	if g.ProcessingTimeHint != "" {
		fmt.Fprintf(&b, "Thời gian: %s\n", g.ProcessingTimeHint)
	}
	if len(g.Notes) > 0 {
		b.WriteString("\nLưu ý:\n")
		for _, note := range g.Notes {
			fmt.Fprintf(&b, "- %s\n", note)
		}
	}
	if len(citations) > 0 {
		b.WriteString("\nNguồn:\n")
		for _, c := range citations {
			fmt.Fprintf(&b, "- %s\n", c.Title)
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

func filledSlots(state map[string]SlotValue, keys []string) map[string]any {
	out := map[string]any{}
	for _, key := range keys {
		slot, ok := state[key]
		if ok && slot.Filled() {
			out[key] = slot.Value
		}
	}
	return out
}

func cloneState(in map[string]SlotValue) map[string]SlotValue {
	out := make(map[string]SlotValue, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

func containsStr(list []string, key string) bool {
	for _, item := range list {
		if item == key {
			return true
		}
	}
	return false
}

// OutOfScopeReply is the fixed citizen reply when no procedure matches.
// V1 citizen scope is one domain: hộ tịch và chứng thực.
func OutOfScopeReply(procedureNames []string) string {
	text := "Hiện trợ lý chỉ hỗ trợ thủ tục hộ tịch và chứng thực của xã."
	if len(procedureNames) == 0 {
		return text
	}
	return text + "\n\nBạn có thể hỏi về: " + strings.Join(procedureNames, ", ") + "."
}
