package drafting

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"sort"
	"strings"
	"unicode"

	"github.com/google/uuid"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
)

// Aggregation merges per-batch partial Definitions into one Definition without
// a model call, so the result is reproducible from the stored batch results.
//
// Rules:
//   - slots merge by key (enum values are unioned); required_slots only carry
//     evidence quoted for "required:<key>", never the slot's own evidence;
//   - checklist/notes/conditional rules are de-duplicated only when their text
//     is identical after normalising case, punctuation and spacing. No fuzzy
//     matching: "Không thu lệ phí" and "Thu lệ phí" differ by one word and mean
//     the opposite, so near-identical wording is never treated as equivalent;
//   - legal scalar fields (where_to_submit, fee_hint, processing_time_hint) with
//     different values in different batches become an OPEN conflict and stay
//     empty until an Admin chooses;
//   - the same slot key proposed with a different type, question or answer
//     options is an OPEN conflict (it decides how Citizen Chat asks);
//   - items beyond a Definition list limit are recorded as Truncated, never
//     dropped silently;
//   - every merged proposal keeps all supporting quotations as provenance.

const (
	aggMaxSlots       = 30
	aggMaxChecklist   = 40
	aggMaxNotes       = 20
	aggMaxIntents     = 20
	aggMaxConditional = 20
	aggMaxCitations   = 10
)

// legalScalarFields are the guidance fields a source can contradict itself on.
var legalScalarFields = []string{"where_to_submit", "fee_hint", "processing_time_hint"}

type aggSlot struct {
	Type       string   `json:"type"`
	Question   string   `json:"question"`
	EnumValues []string `json:"enum_values,omitempty"`
	EnumLabels []string `json:"enum_labels,omitempty"`
}

type aggCond struct {
	When struct {
		Slot   string          `json:"slot"`
		Equals json.RawMessage `json:"equals"`
	} `json:"when"`
	Require []string `json:"require"`
}

type aggGuidance struct {
	Summary              string            `json:"summary"`
	Checklist            []string          `json:"checklist"`
	WhereToSubmit        string            `json:"where_to_submit"`
	Notes                []string          `json:"notes"`
	FeeHint              string            `json:"fee_hint"`
	ProcessingTimeHint   string            `json:"processing_time_hint"`
	ConditionalChecklist []aggGuidanceItem `json:"conditional_checklist,omitempty"`
	ConditionalNotes     []aggGuidanceItem `json:"conditional_notes,omitempty"`
}

type aggGuidanceItem struct {
	When struct {
		Slot   string          `json:"slot"`
		Equals json.RawMessage `json:"equals"`
	} `json:"when"`
	Item string `json:"item"`
}

// guidanceItemKey: same condition (slot and answer) and same text.
func guidanceItemKey(item aggGuidanceItem) string {
	var equals any
	_ = json.Unmarshal(item.When.Equals, &equals)
	canonical, _ := json.Marshal(equals)
	return item.When.Slot + "=" + string(canonical) + "\n" + normText(item.Item)
}

type aggDef struct {
	ProcedureCode    string             `json:"procedure_code"`
	Name             string             `json:"name"`
	IntentExamples   []string           `json:"intent_examples"`
	Slots            map[string]aggSlot `json:"slots"`
	RequiredSlots    []string           `json:"required_slots"`
	ConditionalSlots []aggCond          `json:"conditional_slots"`
	Guidance         aggGuidance        `json:"guidance"`
	Citations        []json.RawMessage  `json:"citations"`
}

func (g aggGuidance) scalar(field string) string {
	switch field {
	case "where_to_submit":
		return g.WhereToSubmit
	case "fee_hint":
		return g.FeeHint
	case "processing_time_hint":
		return g.ProcessingTimeHint
	case "summary":
		return g.Summary
	}
	return ""
}

func (g *aggGuidance) setScalar(field, value string) {
	switch field {
	case "where_to_submit":
		g.WhereToSubmit = value
	case "fee_hint":
		g.FeeHint = value
	case "processing_time_hint":
		g.ProcessingTimeHint = value
	case "summary":
		g.Summary = value
	}
}

// Contribution is one input of the merge: the base draft (SUPPLEMENT) or a batch.
type Contribution struct {
	Label     string
	PageStart int
	PageEnd   int
	// Definition is Go-format Definition JSON; Evidence uses proposal IDs of that Definition.
	Definition json.RawMessage
	Evidence   []EvidenceItem
	Warnings   []string
	Missing    []string
	// OpenConflicts (base only) are carried into the merge as extra candidates.
	OpenConflicts []DraftConflict
	// Prefixed: warnings/missing already carry their page label.
	Prefixed bool
}

type AggregateResult struct {
	Definition        json.RawMessage
	Evidence          []EvidenceItem
	Provenance        []ProposalProvenance
	Conflicts         []DraftConflict
	Warnings          []string
	Missing           []string
	DuplicatesRemoved int
	Truncated         []Truncation
}

// truncationBlocks: lists whose loss changes what a citizen is asked or told.
// Citations block too: which document references to keep is an Admin choice,
// never "the first ten in batch order". Intent examples only tune matching.
var truncationBlocks = map[string]bool{
	"slots": true, "conditional_slots": true, "guidance.checklist": true, "guidance.notes": true, "citations": true,
	"guidance.conditional_checklist": true, "guidance.conditional_notes": true,
}

type contribution struct {
	Contribution
	def      aggDef
	props    map[string]MaterialProposal
	evidence map[string][]EvidenceItem
}

type evRef struct {
	c  *contribution
	id string
}

type textGroup struct {
	text string
	refs []evRef
	// all member texts, for similarity against later items
	members []string
}

// Aggregate merges contributions in order (base first, then batches).
func Aggregate(procedureCode, procedureName string, contributions []Contribution, excerpts []Excerpt) (AggregateResult, error) {
	out := AggregateResult{}
	byExcerpt := make(map[string]Excerpt, len(excerpts))
	for _, ex := range excerpts {
		byExcerpt[ex.ExcerptID.String()] = ex
	}
	cs := make([]*contribution, 0, len(contributions))
	for _, in := range contributions {
		c := &contribution{Contribution: in, evidence: map[string][]EvidenceItem{}}
		if err := json.Unmarshal(in.Definition, &c.def); err != nil {
			return out, fmt.Errorf("drafting: %s definition: %w", in.Label, err)
		}
		props, err := EnumerateMaterialProposals(in.Definition)
		if err != nil {
			// A batch that found nothing parseable contributes no proposals.
			props = nil
		}
		c.props = proposalsByID(props)
		for _, ev := range in.Evidence {
			id := ev.ProposalID
			if norm := NormalizeProposalID(firstNonEmpty(ev.RawProposalID, ev.ProposalID), c.props); !norm.Unknown && !norm.Ambiguous {
				id = norm.Canonical
			}
			c.evidence[id] = append(c.evidence[id], ev)
		}
		cs = append(cs, c)
	}

	merged := aggDef{
		ProcedureCode: procedureCode,
		Name:          procedureName,
		Slots:         map[string]aggSlot{},
	}
	refs := map[string][]evRef{} // merged proposal id -> supporting refs

	// Slots, in order of first appearance.
	slotOrder := make([]string, 0)
	slotRefs := map[string][]evRef{}
	slotCands := map[string][]ConflictCandidate{}
	candRefs := map[string]map[string][]evRef{} // slot key -> signature -> refs
	slotCandidate := func(c *contribution, key string, slot aggSlot) ConflictCandidate {
		if candRefs[key] == nil {
			candRefs[key] = map[string][]evRef{}
		}
		candRefs[key][slotSignature(slot)] = []evRef{{c, SlotProposalID(key)}}
		raw, _ := json.Marshal(slot)
		cand := ConflictCandidate{Value: describeSlot(slot), Contributor: c.Label, PageStart: c.PageStart, PageEnd: c.PageEnd, Slot: raw}
		if ev, ok := chooseEvidence([]evRef{{c, SlotProposalID(key)}}, byExcerpt); ok {
			ev.ProposalID, ev.RawProposalID = SlotProposalID(key), SlotProposalID(key)
			cand.Evidence = &ev
		}
		return cand
	}
	truncated := map[string][]string{}
	for _, c := range cs {
		for _, key := range sortedSlotKeys(c.def.Slots) {
			slot := c.def.Slots[key]
			cur, ok := merged.Slots[key]
			if !ok {
				if len(slotOrder) >= aggMaxSlots {
					truncated["slots"] = append(truncated["slots"], key+": "+describeSlot(slot))
					continue
				}
				merged.Slots[key] = slot
				slotOrder = append(slotOrder, key)
				slotRefs[key] = []evRef{{c, SlotProposalID(key)}}
				slotCands[key] = []ConflictCandidate{slotCandidate(c, key, slot)}
				continue
			}
			out.DuplicatesRemoved++
			if slotSignature(cur) == slotSignature(slot) {
				slotRefs[key] = append(slotRefs[key], evRef{c, SlotProposalID(key)})
				// Labels are display text: take a later batch's when the first
				// had none and the values come in the same order.
				if len(cur.EnumLabels) == 0 && len(slot.EnumLabels) == len(slot.EnumValues) && slices.Equal(cur.EnumValues, slot.EnumValues) {
					cur.EnumLabels = slot.EnumLabels
					merged.Slots[key] = cur
				}
				continue
			}
			// Same key, different type/question/options: keep the first in the
			// JSON but block approval until an Admin picks one (OPEN conflict).
			if r, known := candRefs[key][slotSignature(slot)]; known {
				candRefs[key][slotSignature(slot)] = append(r, evRef{c, SlotProposalID(key)})
			} else {
				slotCands[key] = append(slotCands[key], slotCandidate(c, key, slot))
			}
		}
	}
	for key, r := range slotRefs {
		refs[SlotProposalID(key)] = r
	}
	for _, key := range slotOrder {
		if len(slotCands[key]) > 1 {
			for i := range slotCands[key] {
				var have aggSlot
				_ = json.Unmarshal(slotCands[key][i].Slot, &have)
				r := candRefs[key][slotSignature(have)]
				if i == 0 {
					r = slotRefs[key]
				}
				slotCands[key][i].Sources = sourcesOf(r, byExcerpt)
				// Main evidence: the first valid quotation among ALL batches
				// that proposed this option, not only the first one.
				if ev, ok := chooseEvidence(r, byExcerpt); ok {
					ev.ProposalID, ev.RawProposalID = SlotProposalID(key), SlotProposalID(key)
					slotCands[key][i].Evidence = &ev
				}
			}
			out.Conflicts = append(out.Conflicts, DraftConflict{
				ID: SlotProposalID(key), Field: "slots." + key, Status: ConflictOpen, Candidates: slotCands[key],
			})
		}
	}

	// Required slots: only keys a contribution itself marked required.
	requiredSeen := map[string]bool{}
	for _, c := range cs {
		for _, key := range c.def.RequiredSlots {
			if _, ok := merged.Slots[key]; !ok {
				continue
			}
			id := RequiredProposalID(key)
			if requiredSeen[key] {
				out.DuplicatesRemoved++
			} else {
				requiredSeen[key] = true
				merged.RequiredSlots = append(merged.RequiredSlots, key)
			}
			refs[id] = append(refs[id], evRef{c, id})
		}
	}

	// Conditional rules, de-duplicated by meaning.
	condIndex := map[string]int{}
	condRefs := make([][]evRef, 0)
	for _, c := range cs {
		for i, rule := range c.def.ConditionalSlots {
			if _, ok := merged.Slots[rule.When.Slot]; !ok {
				continue
			}
			key := conditionalKey(rule)
			if idx, ok := condIndex[key]; ok {
				out.DuplicatesRemoved++
				condRefs[idx] = append(condRefs[idx], evRef{c, ConditionalProposalID(i)})
				continue
			}
			if len(merged.ConditionalSlots) >= aggMaxConditional {
				truncated["conditional_slots"] = append(truncated["conditional_slots"], key)
				continue
			}
			condIndex[key] = len(merged.ConditionalSlots)
			merged.ConditionalSlots = append(merged.ConditionalSlots, rule)
			condRefs = append(condRefs, []evRef{{c, ConditionalProposalID(i)}})
		}
	}
	for i, r := range condRefs {
		refs[ConditionalProposalID(i)] = r
	}

	// Checklist and notes.
	checklist := make([]*textGroup, 0)
	notes := make([]*textGroup, 0)
	for _, c := range cs {
		for i, item := range c.def.Guidance.Checklist {
			switch addToGroups(&checklist, item, evRef{c, ChecklistProposalID(i)}, aggMaxChecklist) {
			case groupDuplicate:
				out.DuplicatesRemoved++
			case groupDropped:
				truncated["guidance.checklist"] = append(truncated["guidance.checklist"], item)
			}
		}
		for _, item := range c.def.Guidance.Notes {
			switch addToGroups(&notes, item, evRef{}, aggMaxNotes) {
			case groupDuplicate:
				out.DuplicatesRemoved++
			case groupDropped:
				truncated["guidance.notes"] = append(truncated["guidance.notes"], item)
			}
		}
	}
	merged.Guidance.Checklist = make([]string, 0, len(checklist))
	for i, g := range checklist {
		merged.Guidance.Checklist = append(merged.Guidance.Checklist, g.text)
		refs[ChecklistProposalID(i)] = g.refs
	}
	merged.Guidance.Notes = make([]string, 0, len(notes))
	for _, g := range notes {
		merged.Guidance.Notes = append(merged.Guidance.Notes, g.text)
	}

	// Conditional checklist items and notes: de-duplicated by condition and
	// exact text. An item whose question is not in the merged Definition
	// cannot be shown to anyone; it is reported instead of kept.
	condItems := func(field string, pick func(aggGuidance) []aggGuidanceItem, idOf func(int) string, limit int) []aggGuidanceItem {
		kept := make([]aggGuidanceItem, 0)
		index := map[string]int{}
		itemRefs := make([][]evRef, 0)
		for _, c := range cs {
			for i, item := range pick(c.def.Guidance) {
				if strings.TrimSpace(item.Item) == "" {
					continue
				}
				if _, ok := merged.Slots[item.When.Slot]; !ok {
					out.Warnings = appendUnique(out.Warnings, fmt.Sprintf("Bỏ mục có điều kiện «%s»: câu hỏi %s không có trong bản gộp.", strings.TrimSpace(item.Item), item.When.Slot))
					continue
				}
				key := guidanceItemKey(item)
				if at, ok := index[key]; ok {
					out.DuplicatesRemoved++
					if idOf != nil {
						itemRefs[at] = append(itemRefs[at], evRef{c, idOf(i)})
					}
					continue
				}
				if len(kept) >= limit {
					truncated[field] = append(truncated[field], item.When.Slot+"="+string(item.When.Equals)+": "+item.Item)
					continue
				}
				index[key] = len(kept)
				kept = append(kept, item)
				if idOf != nil {
					itemRefs = append(itemRefs, []evRef{{c, idOf(i)}})
				} else {
					itemRefs = append(itemRefs, nil)
				}
			}
		}
		if idOf != nil {
			for i, r := range itemRefs {
				refs[idOf(i)] = r
			}
		}
		return kept
	}
	merged.Guidance.ConditionalChecklist = condItems("guidance.conditional_checklist",
		func(g aggGuidance) []aggGuidanceItem { return g.ConditionalChecklist }, ConditionalChecklistProposalID, aggMaxChecklist)
	merged.Guidance.ConditionalNotes = condItems("guidance.conditional_notes",
		func(g aggGuidance) []aggGuidanceItem { return g.ConditionalNotes }, nil, aggMaxNotes)

	// Summary is descriptive: the first contribution that has one.
	for _, c := range cs {
		if v := strings.TrimSpace(c.def.Guidance.Summary); v != "" {
			merged.Guidance.Summary = v
			refs[GuidanceProposalID("summary")] = []evRef{{c, GuidanceProposalID("summary")}}
			break
		}
	}

	// Legal scalar fields: one value, or an OPEN conflict.
	for _, field := range legalScalarFields {
		id := GuidanceProposalID(field)
		groups := make([]*textGroup, 0)
		extra := make([]ConflictCandidate, 0)
		for _, c := range cs {
			if v := strings.TrimSpace(c.def.Guidance.scalar(field)); v != "" {
				addToGroups(&groups, v, evRef{c, id}, 0)
			}
			for _, oc := range c.OpenConflicts {
				if oc.ID == id && oc.Status != ConflictResolved {
					extra = append(extra, oc.Candidates...)
				}
			}
		}
		for _, cand := range extra {
			if cand.Value != "" && !groupsContain(groups, cand.Value) {
				groups = append(groups, &textGroup{text: cand.Value, members: []string{cand.Value}})
			}
		}
		switch {
		case len(groups) == 1 && len(groups[0].refs) > 0:
			merged.Guidance.setScalar(field, groups[0].text)
			refs[id] = groups[0].refs
		case len(groups) >= 1:
			conflict := DraftConflict{ID: id, Field: "guidance." + field, Status: ConflictOpen}
			for _, g := range groups {
				cand := ConflictCandidate{Value: g.text, Contributor: "base"}
				if len(g.refs) > 0 {
					rc := g.refs[0].c
					cand.Contributor, cand.PageStart, cand.PageEnd = rc.Label, rc.PageStart, rc.PageEnd
					if ev, ok := chooseEvidence(g.refs, byExcerpt); ok {
						ev.ProposalID, ev.RawProposalID = id, id
						cand.Evidence = &ev
					}
					cand.Sources = sourcesOf(g.refs, byExcerpt)
				} else {
					for _, e := range extra {
						if e.Value == g.text {
							cand = e
							break
						}
					}
				}
				conflict.Candidates = append(conflict.Candidates, cand)
			}
			out.Conflicts = append(out.Conflicts, conflict)
		}
	}

	// Intent examples and citations.
	droppedCitationSeen := map[string]bool{}
	droppedCitations := make([]json.RawMessage, 0)
	// One document may be described differently by different batches (title,
	// number, issuer, date). Every distinct description is kept as a candidate;
	// which one is right is an Admin choice, never "the first batch".
	citeVariants := map[string][]ConflictCandidate{}
	citeOrder := make([]string, 0)
	addCiteVariant := func(cand ConflictCandidate) {
		id := citationKey(cand.Citation)
		meta := citationMeta(cand.Citation)
		if _, ok := citeVariants[id]; !ok {
			citeOrder = append(citeOrder, id)
		}
		for i, have := range citeVariants[id] {
			if citationMeta(have.Citation) == meta {
				who := append([]string{have.Contributor}, have.AlsoFrom...)
				for _, w := range append([]string{cand.Contributor}, cand.AlsoFrom...) {
					if !slices.Contains(who, w) {
						who = append(who, w)
						citeVariants[id][i].AlsoFrom = append(citeVariants[id][i].AlsoFrom, w)
					}
				}
				return
			}
		}
		citeVariants[id] = append(citeVariants[id], cand)
	}
	for _, c := range cs {
		merged.IntentExamples = unionStrings(merged.IntentExamples, c.def.IntentExamples)
		for _, raw := range c.def.Citations {
			addCiteVariant(ConflictCandidate{
				Value: citationMetaLabel(raw), Contributor: c.Label, PageStart: c.PageStart, PageEnd: c.PageEnd, Citation: raw,
			})
		}
		for _, oc := range c.OpenConflicts {
			if oc.Field == "citations" && oc.Status != ConflictResolved {
				for _, cand := range oc.Candidates {
					if len(cand.Citation) > 0 {
						addCiteVariant(cand)
					}
				}
			}
		}
		for _, raw := range c.def.Citations {
			dup := false
			for _, have := range merged.Citations {
				if citationKey(have) == citationKey(raw) {
					dup = true
					break
				}
			}
			switch {
			case dup:
			case len(merged.Citations) >= aggMaxCitations:
				if !droppedCitationSeen[citationKey(raw)] {
					droppedCitationSeen[citationKey(raw)] = true
					truncated["citations"] = append(truncated["citations"], citationLabel(raw))
					droppedCitations = append(droppedCitations, raw)
				}
			default:
				merged.Citations = append(merged.Citations, raw)
			}
		}
	}
	// Conflicts are kept for merge-dropped documents too: they stay inactive
	// until an Admin cites the document again (KeepCitation), then block.
	for _, id := range citeOrder {
		if len(citeVariants[id]) >= 2 {
			out.Conflicts = append(out.Conflicts, DraftConflict{
				ID: citationConflictID(id), Field: "citations", Status: ConflictOpen, Candidates: citeVariants[id],
			})
		}
	}
	if len(merged.IntentExamples) > aggMaxIntents {
		truncated["intent_examples"] = append(truncated["intent_examples"], merged.IntentExamples[aggMaxIntents:]...)
		merged.IntentExamples = merged.IntentExamples[:aggMaxIntents]
	}
	limits := map[string]int{
		"slots": aggMaxSlots, "conditional_slots": aggMaxConditional, "guidance.checklist": aggMaxChecklist,
		"guidance.notes": aggMaxNotes, "citations": aggMaxCitations, "intent_examples": aggMaxIntents,
		"guidance.conditional_checklist": aggMaxChecklist, "guidance.conditional_notes": aggMaxNotes,
	}
	for _, field := range []string{"slots", "conditional_slots", "guidance.checklist", "guidance.conditional_checklist", "guidance.notes", "guidance.conditional_notes", "citations", "intent_examples"} {
		if dropped := truncated[field]; len(dropped) > 0 {
			t := Truncation{Field: field, Limit: limits[field], Dropped: dropped, Blocking: truncationBlocks[field]}
			if field == "citations" {
				t.DroppedCitations = droppedCitations
			}
			out.Truncated = append(out.Truncated, t)
		}
	}
	if merged.IntentExamples == nil {
		merged.IntentExamples = []string{}
	}
	if merged.RequiredSlots == nil {
		merged.RequiredSlots = []string{}
	}
	if merged.ConditionalSlots == nil {
		merged.ConditionalSlots = []aggCond{}
	}
	if merged.Citations == nil {
		merged.Citations = []json.RawMessage{}
	}

	raw, err := json.Marshal(merged)
	if err != nil {
		return out, err
	}
	out.Definition = raw

	// One evidence item per merged proposal; every supporting quote in provenance.
	mergedProps, err := EnumerateMaterialProposals(raw)
	if err != nil {
		return out, fmt.Errorf("drafting: merged definition: %w", err)
	}
	for _, p := range mergedProps {
		r := refs[p.ID]
		out.Provenance = append(out.Provenance, ProposalProvenance{ProposalID: p.ID, Content: p.Content, Sources: sourcesOf(r, byExcerpt)})
		ev, ok := chooseEvidence(r, byExcerpt)
		if !ok {
			continue
		}
		ev.ProposalID, ev.RawProposalID = p.ID, p.ID
		ev.BoundFingerprint, ev.ProposalFingerprint = "", ""
		ev.Status, ev.ReviewWarning, ev.QuoteValid = "", "", false
		ev.ReviewStatus = keptReview(r, ev, p)
		out.Evidence = append(out.Evidence, ev)
	}

	for _, c := range cs {
		label := ""
		if !c.Prefixed {
			label = fmt.Sprintf("[trang %d–%d] ", c.PageStart, c.PageEnd)
		}
		for _, w := range c.Warnings {
			out.Warnings = appendUnique(out.Warnings, label+strings.TrimSpace(w))
		}
		for _, m := range c.Missing {
			out.Missing = appendUnique(out.Missing, label+strings.TrimSpace(m))
		}
	}
	if out.Warnings == nil {
		out.Warnings = []string{}
	}
	if out.Missing == nil {
		out.Missing = []string{}
	}
	return out, nil
}

// keptReview keeps an Admin "accepted" decision only when the evidence comes
// from the base draft and the merged proposal text is unchanged.
func keptReview(r []evRef, chosen EvidenceItem, merged MaterialProposal) string {
	for _, ref := range r {
		if ref.c.Label != "base" {
			continue
		}
		prev, ok := ref.c.props[ref.id]
		if !ok || prev.Content != merged.Content {
			return ""
		}
		for _, ev := range ref.c.evidence[ref.id] {
			if ev.ExcerptID == chosen.ExcerptID && ev.Quotation == chosen.Quotation && ev.ReviewStatus == ReviewAccepted {
				return ReviewAccepted
			}
		}
	}
	return ""
}

// chooseEvidence returns the first quotation found verbatim in its excerpt.
func chooseEvidence(r []evRef, byExcerpt map[string]Excerpt) (EvidenceItem, bool) {
	var first *EvidenceItem
	for _, ref := range r {
		for _, ev := range ref.c.evidence[ref.id] {
			if first == nil {
				e := ev
				first = &e
			}
			if quoteHolds(ev, byExcerpt) {
				return ev, true
			}
		}
	}
	if first != nil {
		return *first, true
	}
	return EvidenceItem{}, false
}

func quoteHolds(ev EvidenceItem, byExcerpt map[string]Excerpt) bool {
	ex, ok := byExcerpt[strings.ToLower(ev.ExcerptID)]
	if !ok {
		return false
	}
	all := make([]Excerpt, 0, len(byExcerpt))
	for _, item := range byExcerpt {
		all = append(all, item)
	}
	return newExcerptIndex(all).quoteFound(ev.Quotation, ev.Page, ex)
}

func sourcesOf(r []evRef, byExcerpt map[string]Excerpt) []ProvenanceRef {
	out := make([]ProvenanceRef, 0)
	for _, ref := range r {
		for _, ev := range ref.c.evidence[ref.id] {
			out = append(out, provenanceRef(ref.c.Label, ev, byExcerpt))
		}
	}
	return out
}

func provenanceRef(label string, ev EvidenceItem, byExcerpt map[string]Excerpt) ProvenanceRef {
	ref := ProvenanceRef{Contributor: label, ExcerptID: ev.ExcerptID, Page: ev.Page, Quotation: ev.Quotation, ChunkIndex: -1}
	if ex, ok := byExcerpt[strings.ToLower(ev.ExcerptID)]; ok {
		ref.ChunkIndex = ex.ChunkIndex
	}
	ref.QuoteValid = quoteHolds(ev, byExcerpt)
	return ref
}

type groupResult int

const (
	groupAdded groupResult = iota
	groupDuplicate
	groupDropped
	groupEmpty
)

// addToGroups adds text to the group with the same normalised text. Wording
// that differs in any word is a separate group: legal meaning can flip on one
// word ("không", a number, a deadline). A new item beyond max is reported as
// dropped so the caller can record it.
func addToGroups(groups *[]*textGroup, text string, ref evRef, max int) groupResult {
	text = strings.TrimSpace(text)
	if text == "" {
		return groupEmpty
	}
	for _, g := range *groups {
		if sameText(g.text, text) {
			g.members = append(g.members, text)
			if ref.c != nil {
				g.refs = append(g.refs, ref)
			}
			return groupDuplicate
		}
	}
	if max > 0 && len(*groups) >= max {
		return groupDropped
	}
	g := &textGroup{text: text, members: []string{text}}
	if ref.c != nil {
		g.refs = []evRef{ref}
	}
	*groups = append(*groups, g)
	return groupAdded
}

func groupsContain(groups []*textGroup, text string) bool {
	for _, g := range groups {
		if sameText(g.text, text) {
			return true
		}
	}
	return false
}

// sameText: equal after lowercasing and dropping punctuation/extra spaces only.
func sameText(a, b string) bool {
	na := normText(a)
	return na != "" && na == normText(b)
}

func normText(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		} else {
			b.WriteRune(' ')
		}
	}
	return strings.Join(strings.Fields(b.String()), " ")
}

// basisText: case and spacing only. Punctuation is kept: in legal values it is
// meaningful ("163/QĐ-BTP" is not "163-QĐ/BTP", "8.000" is not "8000").
func basisText(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(s)), " ")
}

// slotSignature: two slots are the same question only if type, wording and
// answer options all match (case/punctuation aside).
func slotSignature(slot aggSlot) string {
	opts := make([]string, 0, len(slot.EnumValues))
	for _, v := range slot.EnumValues {
		opts = append(opts, normText(v))
	}
	sort.Strings(opts)
	return slot.Type + "|" + normText(slot.Question) + "|" + strings.Join(opts, ",")
}

// describeSlot renders a slot candidate for the Admin conflict view.
func describeSlot(slot aggSlot) string {
	text := slot.Type + ": " + strings.TrimSpace(slot.Question)
	if len(slot.EnumValues) > 0 {
		text += " [" + strings.Join(slot.EnumValues, ", ") + "]"
	}
	return text
}

func conditionalKey(rule aggCond) string {
	req := append([]string{}, rule.Require...)
	sort.Strings(req)
	return rule.When.Slot + "=" + strings.TrimSpace(string(rule.When.Equals)) + "|" + strings.Join(req, ",")
}

// citationLabel is the Admin-facing text of a citation.
func citationLabel(raw json.RawMessage) string {
	var c struct {
		Title string `json:"title"`
	}
	_ = json.Unmarshal(raw, &c)
	return strings.TrimSpace(c.Title)
}

// citationMeta: how a citation describes its document (title, number, issuer,
// effective date), normalised for case and spacing only (see basisText).
func citationMeta(raw json.RawMessage) string {
	var c decision.Citation
	_ = json.Unmarshal(raw, &c)
	return strings.Join([]string{basisText(c.Title), basisText(c.DocumentNumber), basisText(c.Issuer), basisText(c.EffectiveDate)}, "|")
}

// citationMetaLabel shows a citation's description to the Admin.
func citationMetaLabel(raw json.RawMessage) string {
	var c decision.Citation
	_ = json.Unmarshal(raw, &c)
	parts := []string{strings.TrimSpace(c.Title)}
	for _, extra := range []struct{ label, v string }{
		{"số", c.DocumentNumber}, {"cơ quan ban hành", c.Issuer}, {"hiệu lực", c.EffectiveDate},
	} {
		if v := strings.TrimSpace(extra.v); v != "" {
			parts = append(parts, extra.label+" "+v)
		}
	}
	return strings.Join(parts, " · ")
}

// citationConflictID names the description conflict of one cited document.
func citationConflictID(identity string) string {
	return "citation:" + hashStrings([]string{identity})
}

// citationKey identifies a cited source (see citationIdentity).
func citationKey(raw json.RawMessage) string {
	return citationIdentity(raw)
}

// citationIdentity: a document UUID (doc_id or document_id — the same document
// may be written either way) identifies the source on its own, so batches that
// word the title differently still cite it once. A free-text ID is not a
// document reference, so number and title stay part of the key. Conflicting
// doc_id/document_id keep both (the validator blocks them as
// CITATION_SOURCE_CONFLICT), so neither is ever ignored.
func citationIdentity(raw json.RawMessage) string {
	var c decision.Citation
	_ = json.Unmarshal(raw, &c)
	docID := decision.NormalizeSourceID(c.DocID)
	documentID := decision.NormalizeSourceID(c.DocumentID)
	source := docID
	switch {
	case docID == "":
		source = documentID
	case documentID != "" && documentID != docID:
		source = docID + "≠" + documentID
	}
	if id, err := uuid.Parse(source); err == nil {
		return "doc:" + id.String() + "|" + strings.TrimSpace(c.SourceType)
	}
	return strings.Join([]string{
		source, normText(c.DocumentNumber), normText(c.Title), strings.TrimSpace(c.SourceType),
	}, "|")
}

func sortedSlotKeys(m map[string]aggSlot) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func unionStrings(a, b []string) []string {
	out := append([]string{}, a...)
	for _, v := range b {
		out = appendUnique(out, v)
	}
	return out
}

func appendUnique(list []string, v string) []string {
	v = strings.TrimSpace(v)
	if v == "" {
		return list
	}
	for _, have := range list {
		if normText(have) == normText(v) {
			return list
		}
	}
	return append(list, v)
}

// coverageTopics are the standard sections of a procedure in a Vietnamese
// administrative-procedure decision and the Definition field each one feeds.
var coverageTopics = []struct {
	topic string
	field string
	re    *regexp.Regexp
}{
	{"Trình tự thực hiện", "guidance.summary", regexp.MustCompile(`(?i)trình\s*tự\s*thực\s*hiện`)},
	{"Cách thức / cơ quan thực hiện", "guidance.where_to_submit", regexp.MustCompile(`(?i)cách\s*thức\s*thực\s*hiện|cơ\s*quan\s*(?:có\s*thẩm\s*quyền\s*)?thực\s*hiện`)},
	{"Thành phần hồ sơ", "guidance.checklist", regexp.MustCompile(`(?i)thành\s*phần\s*,?\s*(?:số\s*lượng\s*)?hồ\s*sơ`)},
	{"Thời hạn giải quyết", "guidance.processing_time_hint", regexp.MustCompile(`(?i)thời\s*hạn\s*giải\s*quyết`)},
	{"Lệ phí", "guidance.fee_hint", regexp.MustCompile(`(?i)lệ\s*phí`)},
}

// CoverageGaps lists standard sections present in the source whose Definition
// field is empty. A field under an OPEN conflict is not a gap: it is reported
// as the conflict instead. A gap the Admin acknowledged (with a reason, e.g.
// "the document states no fee") stays listed but no longer blocks approval.
func CoverageGaps(definition json.RawMessage, excerpts []Excerpt, conflicts []DraftConflict, acks []GapAck) []CoverageGap {
	var def aggDef
	if err := json.Unmarshal(definition, &def); err != nil {
		return []CoverageGap{}
	}
	open := map[string]bool{}
	for _, c := range conflicts {
		if c.Status != ConflictResolved {
			open[c.Field] = true
		}
	}
	out := make([]CoverageGap, 0)
	for _, t := range coverageTopics {
		page := 0
		for _, ex := range excerpts {
			if t.re.MatchString(ex.Content) {
				page = ex.PageStart
				break
			}
		}
		if page == 0 || open[t.field] {
			continue
		}
		var empty bool
		if t.field == "guidance.checklist" {
			empty = len(def.Guidance.Checklist) == 0 && len(def.Guidance.ConditionalChecklist) == 0
		} else {
			empty = strings.TrimSpace(def.Guidance.scalar(strings.TrimPrefix(t.field, "guidance."))) == ""
		}
		if empty {
			gap := CoverageGap{Topic: t.topic, Field: t.field, Page: page, Basis: fmt.Sprintf("gap|%s|%s|%d", t.field, t.topic, page)}
			if a, ok := findAck(acks, t.field, gap.Basis); ok {
				gap.Acknowledged, gap.Reason = true, a.Reason
			}
			out = append(out, gap)
		}
	}
	return out
}

// ackFinding is one coverage finding an Admin may decide on, with the exact
// state (Basis) the decision applies to.
type ackFinding struct {
	Field string
	Basis string
}

func findAck(acks []GapAck, field, basis string) (GapAck, bool) {
	for _, a := range acks {
		if a.Field == field && a.Basis == basis {
			return a, true
		}
	}
	return GapAck{}, false
}

func hashStrings(items []string) string {
	sum := sha256.Sum256([]byte(strings.Join(items, "\n")))
	return hex.EncodeToString(sum[:8])
}

// truncationBasis pins a review to the dropped items. Citations use their
// full source identity (two documents may share a title). Order does not
// matter: the same set gives the same basis. Only a hash is stored.
func truncationBasis(t Truncation) string {
	items := append([]string{}, t.Dropped...)
	if t.Field == "citations" && len(t.DroppedCitations) > 0 {
		items = items[:0]
		for _, raw := range t.DroppedCitations {
			items = append(items, citationIdentity(raw))
		}
	}
	sort.Strings(items)
	return "truncated|" + t.Field + "|" + hashStrings(items)
}

// unsectionedMentions: places that name the procedure outside EVERY procedure
// section (general provisions, tables, appendices). Mentions inside another
// procedure's section belong to that procedure and are not counted.
func unsectionedMentions(c *DraftCoverage) []OutsideMention {
	out := make([]OutsideMention, 0)
	for _, m := range c.OutsideMentions {
		if m.Section == "" {
			out = append(out, m)
		}
	}
	return out
}

// outsideBasis pins a review to the exact chunks found, not page bounds/counts.
func outsideBasis(ms []OutsideMention) string {
	parts := make([]string, 0, len(ms))
	for _, m := range ms {
		ids := make([]string, 0, len(m.ChunkIndexes))
		for _, c := range m.ChunkIndexes {
			ids = append(ids, fmt.Sprint(c))
		}
		parts = append(parts, m.Kind+":"+strings.Join(ids, ";"))
	}
	return "outside|" + hashStrings(parts)
}

// pendingFindings lists every finding that currently needs an Admin decision.
func pendingFindings(c *DraftCoverage) []ackFinding {
	out := make([]ackFinding, 0)
	for _, g := range c.Gaps {
		out = append(out, ackFinding{Field: g.Field, Basis: g.Basis})
	}
	for _, t := range c.Truncated {
		out = append(out, ackFinding{Field: "truncated:" + t.Field, Basis: truncationBasis(t)})
	}
	if ms := unsectionedMentions(c); len(ms) > 0 {
		out = append(out, ackFinding{Field: "outside_mentions", Basis: outsideBasis(ms)})
	}
	if c.BoundaryUncertain {
		out = append(out, ackFinding{Field: "section_boundary", Basis: boundaryBasis(c)})
	}
	return out
}

func boundaryBasis(c *DraftCoverage) string {
	return fmt.Sprintf("boundary|%s|%d-%d", c.SectionTitle, c.PageStart, c.PageEnd)
}

// conflictValueBasis: what the Definition currently says for a conflict's
// field (a citations conflict: how the document is described, empty when it is
// not cited).
func conflictValueBasis(def json.RawMessage, c DraftConflict) string {
	var obj struct {
		Slots     map[string]json.RawMessage `json:"slots"`
		Guidance  map[string]any             `json:"guidance"`
		Citations []json.RawMessage          `json:"citations"`
	}
	_ = json.Unmarshal(def, &obj)
	value := ""
	switch {
	case c.Field == "citations":
		for _, raw := range obj.Citations {
			if citationConflictID(citationIdentity(raw)) == c.ID {
				value = citationMeta(raw)
			}
		}
	case strings.HasPrefix(c.Field, "slots."):
		if raw, ok := obj.Slots[strings.TrimPrefix(c.Field, "slots.")]; ok {
			var slot any
			_ = json.Unmarshal(raw, &slot)
			canonical, _ := json.Marshal(slot)
			value = string(canonical)
		}
	default:
		text, _ := obj.Guidance[strings.TrimPrefix(c.Field, "guidance.")].(string)
		value = basisText(text)
	}
	return hashStrings([]string{c.Field, value})
}

// reconcileConflicts keeps conflict state true to the current Definition: a
// citations conflict is inactive while its document is not cited, and a
// decision whose value was edited afterwards reopens. A decision recorded
// without a basis (before decisions were pinned) cannot be checked, so it
// reopens too (fail closed). A reopened decision loses the provenance it set.
func reconcileConflicts(view *DraftView) {
	cited := map[string]bool{}
	var obj struct {
		Citations []json.RawMessage `json:"citations"`
	}
	_ = json.Unmarshal(view.Definition, &obj)
	for _, raw := range obj.Citations {
		cited[citationConflictID(citationIdentity(raw))] = true
	}
	for i := range view.Conflicts {
		c := &view.Conflicts[i]
		c.Inactive = c.Field == "citations" && !cited[c.ID]
		if c.Status == ConflictResolved && (c.ResolvedBasis == "" || conflictValueBasis(view.Definition, *c) != c.ResolvedBasis) {
			c.Status, c.Chosen, c.ResolvedBy, c.ResolvedAt, c.ResolvedBasis, c.Reopened = ConflictOpen, nil, nil, nil, "", true
			kept := view.Provenance[:0:0]
			for _, p := range view.Provenance {
				if p.ProposalID != c.ID {
					kept = append(kept, p)
				}
			}
			view.Provenance = kept
		}
	}
}

// syncConflictIssues makes the SOURCE_CONFLICT issues match the conflicts:
// one error per open, active conflict and none otherwise.
func syncConflictIssues(view *DraftView) {
	v := &view.Validation
	kept := v.Issues[:0:0]
	for _, is := range v.Issues {
		if is.Code != "SOURCE_CONFLICT" {
			kept = append(kept, is)
		}
	}
	for _, c := range view.Conflicts {
		if c.Status == ConflictResolved || c.Inactive {
			continue
		}
		kept = append(kept, decision.DraftIssue{
			Code:       "SOURCE_CONFLICT",
			Severity:   "error",
			Message:    fmt.Sprintf("source batches disagree on %s (%d values); an Admin must choose", c.Field, len(c.Candidates)),
			ProposalID: c.ID,
		})
	}
	v.Issues = kept
}

// applyAggregateFindings re-adds job-draft findings after every revalidation.
// Open conflicts, unacknowledged coverage gaps, blocking truncations and
// unreviewed outside mentions are errors (they block approval). Acknowledgements
// that no longer match a current finding are dropped, so e.g. a field that was
// filled and later emptied again needs a fresh decision.
func applyAggregateFindings(view *DraftView, excerpts []Excerpt) {
	if view.Coverage == nil && len(view.Conflicts) == 0 {
		return
	}
	reconcileConflicts(view)
	syncConflictIssues(view)
	v := &view.Validation
	cov := view.Coverage
	if cov == nil {
		return
	}
	cov.Gaps = CoverageGaps(view.Definition, excerpts, view.Conflicts, nil)
	current := pendingFindings(cov)
	kept := cov.Acknowledged[:0:0]
	for _, a := range cov.Acknowledged {
		for _, f := range current {
			if a.Field == f.Field && a.Basis == f.Basis {
				kept = append(kept, a)
				break
			}
		}
	}
	cov.Acknowledged = kept
	cov.Gaps = CoverageGaps(view.Definition, excerpts, view.Conflicts, kept)

	for _, g := range cov.Gaps {
		if g.Acknowledged {
			v.Issues = append(v.Issues, decision.DraftIssue{
				Code: "COVERAGE_GAP_ACKNOWLEDGED", Severity: "warning",
				Message: fmt.Sprintf("%s left empty by Admin decision: %s", g.Field, g.Reason),
			})
			continue
		}
		v.Issues = append(v.Issues, decision.DraftIssue{
			Code: "COVERAGE_GAP", Severity: "error",
			Message: fmt.Sprintf("source has \"%s\" (page %d) but %s is empty; fill it from the source or acknowledge why it stays empty", g.Topic, g.Page, g.Field),
		})
	}
	for _, t := range cov.Truncated {
		if a, ok := findAck(kept, "truncated:"+t.Field, truncationBasis(t)); ok {
			v.Issues = append(v.Issues, decision.DraftIssue{
				Code: "CONTENT_TRUNCATED_ACKNOWLEDGED", Severity: "warning",
				Message: fmt.Sprintf("%d %s item(s) over the limit of %d reviewed by Admin: %s", len(t.Dropped), t.Field, t.Limit, a.Reason),
			})
			continue
		}
		severity := "warning"
		if t.Blocking {
			severity = "error"
		}
		v.Issues = append(v.Issues, decision.DraftIssue{
			Code: "CONTENT_TRUNCATED", Severity: severity,
			Message: fmt.Sprintf("%d %s item(s) were dropped by the limit of %d; review them before approval", len(t.Dropped), t.Field, t.Limit),
		})
	}
	ms := unsectionedMentions(cov)
	_, reviewed := findAck(kept, "outside_mentions", outsideBasis(ms))
	cov.OutsideMentionsReviewed = len(ms) == 0 || reviewed
	if !cov.OutsideMentionsReviewed {
		v.Issues = append(v.Issues, decision.DraftIssue{
			Code: "OUTSIDE_SOURCES_UNREVIEWED", Severity: "error",
			Message: fmt.Sprintf("%d place(s) outside every procedure section match the outside search (%s); review them and record why they add nothing, or supplement the draft", len(ms), cov.OutsideCriteria),
		})
	}
	if cov.BoundaryUncertain {
		_, ok := findAck(kept, "section_boundary", boundaryBasis(cov))
		cov.BoundaryReviewed = ok
		if !ok {
			v.Issues = append(v.Issues, decision.DraftIssue{
				Code: "SECTION_BOUNDARY_UNCERTAIN", Severity: "error",
				Message: fmt.Sprintf("the end of section %q (pages %d-%d) was detected with a numbering gap; an Admin must confirm the pages belong to this procedure", cov.SectionTitle, cov.PageStart, cov.PageEnd),
			})
		}
	}
	cov.ScopeComplete = cov.ScopeComplete || cov.Complete
	cov.Complete = cov.ScopeComplete
	if !cov.ScopeComplete {
		v.Issues = append(v.Issues, decision.DraftIssue{
			Code: "SOURCE_PARTIALLY_PROCESSED", Severity: "error",
			Message: fmt.Sprintf("only %d of %d selected chunks were processed", cov.ProcessedChunks, cov.SelectedChunks),
		})
	}
}

// chunkRangesOf turns ordered chunk indexes into inclusive contiguous ranges.
func chunkRangesOf(indexes []int) []ChunkRange {
	if len(indexes) == 0 {
		return nil
	}
	sorted := append([]int{}, indexes...)
	sort.Ints(sorted)
	out := []ChunkRange{{Start: sorted[0], End: sorted[0]}}
	for _, n := range sorted[1:] {
		last := &out[len(out)-1]
		switch {
		case n == last.End || n == last.End+1:
			last.End = n
		default:
			out = append(out, ChunkRange{Start: n, End: n})
		}
	}
	return out
}

func inRanges(n int, ranges []ChunkRange) bool {
	for _, r := range ranges {
		if n >= r.Start && n <= r.End {
			return true
		}
	}
	return false
}

func uuidStrings(ids []uuid.UUID) []string {
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		out = append(out, id.String())
	}
	return out
}
