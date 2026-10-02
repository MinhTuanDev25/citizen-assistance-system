package chat

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/aiextract"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

// turnWithAI is the P2 three-phase turn: a short read-mostly transaction to
// snapshot state and compute a fingerprint, the AI network call STRICTLY
// outside any transaction/row lock, then a write transaction that re-checks
// the fingerprint and falls back to keyword-only behavior on any mismatch,
// timeout, or invalid AI response. All existing idempotency/replay/409
// semantics are preserved because both transactions reuse the exact same
// LockTurn + FindIdempotency + legacy-replay prologue as turnKeywordOnly.
func (s *Service) turnWithAI(ctx context.Context, sessionID, requestID uuid.UUID, message string) (*TurnResult, error) {
	payloadHash := PayloadHash(message)

	// A same-payload duplicate that arrives while a claim is PENDING waits
	// and then replays. A crashed owner leaves an expired lease, which the
	// next attempt steals. Bounded so a stuck row cannot loop forever.
	for attempt := 0; attempt < 8; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		// --- Phase 1: short transaction. Lock is released before any network I/O.
		tx1, err := s.Pool.Begin(ctx)
		if err != nil {
			return nil, err
		}
		snap, replay, err := s.beginTurnPhase(ctx, tx1, sessionID, requestID, message, payloadHash)
		if err != nil {
			tx1.Rollback(ctx)
			return nil, err
		}
		if replay != nil {
			if err := tx1.Commit(ctx); err != nil {
				return nil, err
			}
			return replay, nil
		}

		// A pending CONFIRM_INTENT resolution (yes/no/name) is deterministic and
		// needs no AI — resolve it here without ever calling the extractor
		// and without taking an extractor claim.
		if pending, ok := decodePending(snap.PendingIntent); ok && len(pending.Candidates) > 0 {
			plan, err := handlePendingConfirm(ctx, s.Sessions, tx1, snap, message, pending, catalogNames(snap))
			if err != nil {
				tx1.Rollback(ctx)
				return nil, err
			}
			if plan != nil {
				return s.finishPlan(ctx, tx1, sessionID, requestID, snap, message, payloadHash, plan)
			}
		}

		fingerprint1 := computeFingerprint(snap)
		extractReq, allowedSlotKeys, pinnedCode := buildExtractRequest(snap, requestID, message)
		decision, err := s.Sessions.ResolveExtractClaim(ctx, tx1, sessionID, requestID, payloadHash, s.claimLease())
		if err != nil {
			tx1.Rollback(ctx)
			return nil, err
		}
		switch decision {
		case repository.ClaimConflict:
			tx1.Rollback(ctx)
			return nil, repository.ErrIdempotencyConflict
		case repository.ClaimWait:
			if err := tx1.Commit(ctx); err != nil {
				return nil, err
			}
			replay, ready, err := s.waitForCompletedClaim(ctx, sessionID, requestID, payloadHash)
			if err != nil {
				return nil, err
			}
			if ready {
				return replay, nil
			}
			continue
		case repository.ClaimOwned:
			if err := tx1.Commit(ctx); err != nil {
				return nil, err
			}
		default:
			tx1.Rollback(ctx)
			return nil, fmt.Errorf("aiextract: unknown claim result")
		}

		// --- Phase 2: AI call — NEVER inside a DB transaction/row lock -----
		outcome := s.callExtractor(ctx, extractReq)
		return s.finishAIWrite(ctx, sessionID, requestID, message, payloadHash, fingerprint1, extractReq, allowedSlotKeys, pinnedCode, outcome)
	}
	return nil, fmt.Errorf("aiextract: exhausted claim retries")
}

// finishAIWrite is phase 3: a new transaction re-checks the fingerprint and
// promotes the PENDING claim to a COMPLETE envelope. The session row lock
// is taken here only for the write, never across the network call.
func (s *Service) finishAIWrite(
	ctx context.Context,
	sessionID, requestID uuid.UUID,
	message, payloadHash, fingerprint1 string,
	extractReq aiextract.Request,
	allowedSlotKeys []string,
	pinnedCode string,
	outcome extractOutcome,
) (*TurnResult, error) {
	tx2, err := s.Pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx2.Rollback(ctx)

	snap2, replay2, err := s.beginTurnPhase(ctx, tx2, sessionID, requestID, message, payloadHash)
	if err != nil {
		return nil, err
	}
	if replay2 != nil {
		if err := tx2.Commit(ctx); err != nil {
			return nil, err
		}
		return replay2, nil
	}

	if pending, ok := decodePending(snap2.PendingIntent); ok && len(pending.Candidates) > 0 {
		plan, err := handlePendingConfirm(ctx, s.Sessions, tx2, snap2, message, pending, catalogNames(snap2))
		if err != nil {
			return nil, err
		}
		if plan != nil {
			return s.finishPlan(ctx, tx2, sessionID, requestID, snap2, message, payloadHash, plan)
		}
	}

	fingerprint2 := computeFingerprint(snap2)
	meta := aiMeta{latencyMs: outcome.latencyMs, fallbackReason: outcome.fallbackReason}

	var validated *aiextract.ValidatedProposal
	if fingerprint2 != fingerprint1 {
		meta.fallbackReason = "context_changed"
	} else if outcome.fallbackReason == "" && outcome.response != nil {
		v, verr := aiextract.Validate(
			outcome.response, extractReq.Candidates, pinnedCode, allowedSlotKeys,
			s.AISlotConfidenceMin, currentSlotState(snap2),
		)
		if verr != nil {
			meta.fallbackReason = "invalid_contract"
		} else {
			validated = v
			// Only the validated allowlisted provider/model are observable.
			// The raw model strings are never copied.
			meta.provider = v.Provider
			meta.model = v.Model
		}
	}

	plan, err := planTurnAI(snap2, message, validated, s.effectivePolicy(), &meta)
	if err != nil {
		return nil, err
	}
	return s.finishPlan(ctx, tx2, sessionID, requestID, snap2, message, payloadHash, plan)
}

func (s *Service) claimLease() time.Duration {
	if s.ClaimLease > 0 {
		return s.ClaimLease
	}
	return DefaultClaimLease
}

// waitForCompletedClaim polls outside any row lock until the owner's claim
// is COMPLETE, the payload hash conflicts, or the lease expires (caller may
// then steal it). It never calls the extractor.
func (s *Service) waitForCompletedClaim(ctx context.Context, sessionID, requestID uuid.UUID, payloadHash string) (*TurnResult, bool, error) {
	deadline := time.Now().Add(s.claimLease())
	for {
		if err := ctx.Err(); err != nil {
			return nil, false, err
		}
		if time.Now().After(deadline) {
			return nil, false, nil
		}
		timer := time.NewTimer(40 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, false, ctx.Err()
		case <-timer.C:
		}

		tx, err := s.Pool.Begin(ctx)
		if err != nil {
			return nil, false, err
		}
		state, ok, err := s.Sessions.FindIdempotencyState(ctx, tx, sessionID, requestID)
		tx.Rollback(ctx)
		if err != nil {
			return nil, false, err
		}
		if !ok {
			return nil, false, nil
		}
		if state.PayloadHash != payloadHash {
			return nil, false, repository.ErrIdempotencyConflict
		}
		if state.Status == repository.IdempotencyComplete {
			var replay TurnResult
			if err := json.Unmarshal(state.ResponseJSON, &replay); err != nil {
				return nil, false, err
			}
			replay.IdempotentReplay = true
			return &replay, true, nil
		}
		if state.ExpiresAt == nil || !state.ExpiresAt.After(time.Now()) {
			return nil, false, nil
		}
	}
}

func (s *Service) effectivePolicy() decision.Policy {
	if s.AIPolicySet {
		return s.AIPolicy
	}
	if s.AIPolicy.SelectMin == 0 && s.AIPolicy.ConfirmMin == 0 {
		return decision.DefaultPolicy
	}
	return s.AIPolicy
}

// beginTurnPhase runs the shared lock + idempotency + legacy-replay prologue
// against tx. Returns (snap, nil, nil) for a fresh turn, or (snap, replay,
// nil) when this request_id was already completed (or is a legacy row).
func (s *Service) beginTurnPhase(
	ctx context.Context, tx pgx.Tx, sessionID, requestID uuid.UUID, message, payloadHash string,
) (*repository.TurnSnapshot, *TurnResult, error) {
	snap, err := s.Sessions.LockTurn(ctx, tx, sessionID, s.CitizenDomains)
	if err != nil {
		return nil, nil, err
	}
	if snap.Session.Status != "OPEN" {
		return nil, nil, repository.ErrSessionNotOpen
	}

	state, ok, err := s.Sessions.FindIdempotencyState(ctx, tx, sessionID, requestID)
	if err != nil {
		return nil, nil, err
	}
	if ok {
		if state.PayloadHash != payloadHash {
			// Conflict is decided before any extractor call, including when
			// the existing row is still an in-flight PENDING claim.
			return nil, nil, repository.ErrIdempotencyConflict
		}
		if state.Status == repository.IdempotencyComplete {
			var replay TurnResult
			if err := json.Unmarshal(state.ResponseJSON, &replay); err != nil {
				return nil, nil, err
			}
			replay.IdempotentReplay = true
			return snap, &replay, nil
		}
		// PENDING with the same hash: not a replay and not a fresh turn.
		// The caller claims or waits. Do not fall through to legacy replay.
		return snap, nil, nil
	}

	if legacy, err := s.replayLegacyTurn(ctx, tx, sessionID, requestID, message, payloadHash); err != nil {
		return nil, nil, err
	} else if legacy != nil {
		return snap, legacy, nil
	}
	return snap, nil, nil
}

// extractOutcome is the result of the Phase-2 AI call, before Go-side
// contract re-validation (which happens back inside Phase 3, against a
// possibly-refreshed snapshot).
type extractOutcome struct {
	response       *aiextract.Response
	latencyMs      int64
	fallbackReason string // "" only when response is non-nil and call succeeded
}

func (s *Service) callExtractor(ctx context.Context, req aiextract.Request) extractOutcome {
	start := time.Now()
	resp, err := s.Extractor.Extract(ctx, req)
	latency := time.Since(start).Milliseconds()
	if err != nil {
		reason := "http_error"
		if errors.Is(err, context.DeadlineExceeded) {
			reason = "timeout"
		} else if errors.Is(err, aiextract.ErrInvalidResponse) {
			reason = "invalid_contract"
		} else if errors.Is(err, aiextract.ErrRateLimited) {
			reason = "rate_limited"
		}
		return extractOutcome{latencyMs: latency, fallbackReason: reason}
	}
	return extractOutcome{response: resp, latencyMs: latency}
}

// computeFingerprint binds everything a Phase-2 AI proposal is conditioned
// on: session identity/status/pin, pending intent, prior slot state, and the
// exact catalog (procedure + version ids + definition hashes) offered as
// candidates. If any of this changes between Phase 1 and Phase 3 (a
// concurrent turn, a catalog change, a pin change), the AI proposal is
// discarded and the write transaction falls back to keyword-only handling.
func computeFingerprint(snap *repository.TurnSnapshot) string {
	h := sha256.New()
	h.Write([]byte(snap.Session.ID.String()))
	h.Write([]byte(snap.Session.Status))
	if snap.Session.ActiveProcedureID != nil {
		h.Write([]byte(snap.Session.ActiveProcedureID.String()))
	}
	if snap.Session.ActiveProcedureVersionID != nil {
		h.Write([]byte(snap.Session.ActiveProcedureVersionID.String()))
	}
	h.Write(snap.PendingIntent)
	h.Write(snap.PriorRaw)

	rows := make([]repository.ProcedureDefinitionRow, len(snap.Catalog))
	copy(rows, snap.Catalog)
	sort.Slice(rows, func(i, j int) bool { return rows[i].ProcedureCode < rows[j].ProcedureCode })
	for _, row := range rows {
		h.Write([]byte(row.ProcedureID.String()))
		h.Write([]byte(row.VersionID.String()))
		h.Write([]byte(DefinitionHash(row.Definition)))
	}
	return hex.EncodeToString(h.Sum(nil))
}

func catalogNames(snap *repository.TurnSnapshot) []string {
	out := make([]string, 0, len(snap.Catalog))
	for _, row := range snap.Catalog {
		out = append(out, row.Name)
	}
	return out
}

// buildExtractRequest builds the exact contract payload for POST /v1/extract
// from a locked snapshot. Also returns the allowed_slot_keys list (only
// meaningful when a procedure is pinned) and the pinned procedure code (""
// when nothing is pinned yet) — both are needed again in Phase 3 to
// re-validate the AI response.
func buildExtractRequest(snap *repository.TurnSnapshot, requestID uuid.UUID, message string) (aiextract.Request, []string, string) {
	candidates := make([]aiextract.Candidate, 0, len(snap.Catalog))
	for _, row := range snap.Catalog {
		def, err := decision.ParseDefinition(row.Definition)
		if err != nil {
			continue
		}
		slots := make(map[string]aiextract.SlotSpec, len(def.Slots))
		for key, sd := range def.Slots {
			slots[key] = aiextract.SlotSpec{
				Type:       sd.Type,
				Question:   sd.Question,
				EnumValues: append([]string{}, sd.EnumValues...),
			}
		}
		candidates = append(candidates, aiextract.Candidate{
			ProcedureCode:  row.ProcedureCode,
			Name:           row.Name,
			IntentExamples: append([]string{}, def.IntentExamples...),
			Slots:          slots,
		})
	}

	var pinned *aiextract.PinnedContext
	pinnedCode := ""
	var allowedSlotKeys []string

	if snap.Pinned != nil {
		pinnedCode = snap.Pinned.ProcedureCode
		if def, err := decision.ParseDefinition(snap.Pinned.Definition); err == nil {
			prior, _ := decodeSlotState(snap.PriorRaw)

			seen := map[string]struct{}{}
			addAllowed := func(key string) {
				if _, ok := seen[key]; ok {
					return
				}
				seen[key] = struct{}{}
				allowedSlotKeys = append(allowedSlotKeys, key)
			}
			for _, key := range def.Missing(prior) {
				addAllowed(key)
			}
			// Also allow correcting an already-answered boolean/enum choice
			// (mirrors the keyword engine's HasSlotCorrection behavior).
			for key, slot := range def.Slots {
				if (slot.Type == "boolean" || slot.Type == "enum") && prior[key].Filled() {
					addAllowed(key)
				}
			}

			state := map[string]aiextract.SlotStateEntry{}
			for key, sv := range prior {
				state[key] = aiextract.SlotStateEntry{Value: sv.Value, Status: sv.Status}
			}
			for _, key := range def.EffectiveRequired(prior) {
				if _, ok := state[key]; !ok {
					state[key] = aiextract.SlotStateEntry{Value: nil, Status: decision.StatusMissing}
				}
			}
			pinned = &aiextract.PinnedContext{
				ProcedureCode:   pinnedCode,
				AllowedSlotKeys: allowedSlotKeys,
				SlotState:       state,
			}
		}
	}

	req := aiextract.Request{
		SchemaVersion: aiextract.SchemaVersion,
		RequestID:     requestID.String(),
		Message:       message,
		Candidates:    candidates,
		PinnedContext: pinned,
	}
	return req, allowedSlotKeys, pinnedCode
}

func currentSlotState(snap *repository.TurnSnapshot) map[string]aiextract.CurrentSlot {
	out := map[string]aiextract.CurrentSlot{}
	if snap.Pinned == nil {
		return out
	}
	prior, _ := decodeSlotState(snap.PriorRaw)
	for key, sv := range prior {
		out[key] = aiextract.CurrentSlot{Filled: sv.Filled()}
	}
	return out
}

// aiMeta carries non-PII observability fields into SaveTurnInput.Metadata.
// Never carries the citizen message, the raw provider response, or any
// secret.
type aiMeta struct {
	source         string // "ai" | "keyword_fallback"
	provider       string
	model          string
	latencyMs      int64
	fallbackReason string
	accepted       []string
	rejected       []string
}

// safeObservability drops anything that is not an allowlisted provider and a
// token-shaped model id. Rejected values are omitted, never stored.
func safeObservability(provider, model string) (string, string) {
	p, m, err := aiextract.SafeProviderModel(provider, model)
	if err != nil {
		return "", ""
	}
	return p, m
}

func attachExtractMetadata(plan *planned, meta *aiMeta) {
	if plan == nil || meta == nil {
		return
	}
	if plan.save.Metadata == nil {
		plan.save.Metadata = map[string]any{}
	}
	source := meta.source
	if source == "" {
		source = "keyword_fallback"
	}
	plan.save.Metadata["extract_source"] = source
	provider, model := safeObservability(meta.provider, meta.model)
	if provider != "" {
		plan.save.Metadata["extract_provider"] = provider
	}
	if model != "" {
		plan.save.Metadata["extract_model"] = model
	}
	plan.save.Metadata["extract_latency_ms"] = meta.latencyMs
	if meta.fallbackReason != "" {
		plan.save.Metadata["extract_fallback_reason"] = meta.fallbackReason
	}
	if len(meta.accepted) > 0 {
		plan.save.Metadata["accepted_slot_keys"] = meta.accepted
	}
	if len(meta.rejected) > 0 {
		plan.save.Metadata["rejected_slot_keys"] = meta.rejected
	}
}

// planTurnAI mirrors planTurn's control flow, substituting an AI-derived
// slot/intent proposal where the keyword engine would otherwise be the only
// signal. It never changes the outcome shape: every branch still terminates
// in a *planned built by buildPlannedFromDecision, confirmIntentFromPending,
// or outOfScope — the exact same building blocks planTurn uses.
func planTurnAI(
	snap *repository.TurnSnapshot,
	message string,
	validated *aiextract.ValidatedProposal,
	policy decision.Policy,
	meta *aiMeta,
) (*planned, error) {
	names := make([]string, 0, len(snap.Catalog))
	candidates := make([]decision.IntentCandidate, 0, len(snap.Catalog))
	byCode := map[string]repository.ProcedureDefinitionRow{}
	for _, row := range snap.Catalog {
		def, err := decision.ParseDefinition(row.Definition)
		if err != nil {
			return nil, fmt.Errorf("procedure %s: %w", row.ProcedureCode, err)
		}
		names = append(names, row.Name)
		candidates = append(candidates, decision.IntentCandidate{
			ProcedureCode: row.ProcedureCode,
			Name:          row.Name,
			Examples:      def.IntentExamples,
		})
		byCode[row.ProcedureCode] = row
	}

	var plan *planned
	var err error
	if snap.Pinned != nil {
		plan, err = planPinnedWithAI(snap, message, candidates, byCode, names, validated, policy, meta)
	} else {
		plan, err = planUnpinnedWithAI(snap, message, candidates, byCode, names, validated, policy, meta)
	}
	if err != nil || plan == nil {
		return plan, err
	}
	attachExtractMetadata(plan, meta)
	return plan, nil
}

func planPinnedWithAI(
	snap *repository.TurnSnapshot,
	message string,
	candidates []decision.IntentCandidate,
	byCode map[string]repository.ProcedureDefinitionRow,
	names []string,
	validated *aiextract.ValidatedProposal,
	policy decision.Policy,
	meta *aiMeta,
) (*planned, error) {
	match := decision.MatchIntentBands(message, candidates)

	// A strong keyword match to a DIFFERENT procedure switches immediately —
	// this is pre-existing keyword behavior (planTurn), left untouched. AI
	// slots are irrelevant here since prior is reset to {} for a fresh pin.
	if match.Band == decision.IntentSelect && match.Selected.ProcedureCode != snap.Pinned.ProcedureCode {
		row := byCode[match.Selected.ProcedureCode]
		if meta.source == "" {
			meta.source = "keyword_fallback"
		}
		return decidePinnedWithExtraction(snap, message, row, map[string]decision.SlotValue{}, false, nil, meta)
	}

	// Stay pinned (or let the AI/keyword propose confirmation).
	row := *snap.Pinned
	prior, err := decodeSlotState(snap.PriorRaw)
	if err != nil {
		return nil, err
	}
	pinnedDef, err := decision.ParseDefinition(row.Definition)
	if err != nil {
		return nil, fmt.Errorf("procedure %s: %w", row.ProcedureCode, err)
	}

	// AI proposes a procedure different from the one pinned: this must
	// always go through CONFIRM_INTENT — never a silent switch, regardless
	// of how confident the AI is.
	if validated != nil && validated.IntentProcedureCode != "" && validated.IntentProcedureCode != row.ProcedureCode {
		if policy.Band(validated.IntentConfidence) != decision.IntentNone {
			meta.source = "ai"
			return confirmIntentFromAI(snap, message, validated, byCode), nil
		}
		// Low-confidence alternate suggestion: ignore the intent-switch
		// signal only; slots targeting the pinned procedure are unaffected.
		validated.IntentProcedureCode = ""
	}

	var extraction *decision.ExtractionProposal
	if validated != nil && validated.SlotsProcedureCode == row.ProcedureCode && len(validated.Slots) > 0 {
		extraction = toDecisionProposal(validated)
		meta.source = "ai"
	}
	if meta.source == "" {
		meta.source = "keyword_fallback"
	}

	hasExtractionSlots := extraction != nil && len(extraction.Slots) > 0
	if match.Band != decision.IntentSelect &&
		len(pinnedDef.Missing(prior)) == 0 &&
		!decision.HasSlotCorrection(pinnedDef, prior, message) &&
		!hasExtractionSlots {
		return outOfScope(snap, message, names), nil
	}
	return decidePinnedWithExtraction(snap, message, row, prior, snap.HadPrior, extraction, meta)
}

func planUnpinnedWithAI(
	snap *repository.TurnSnapshot,
	message string,
	candidates []decision.IntentCandidate,
	byCode map[string]repository.ProcedureDefinitionRow,
	names []string,
	validated *aiextract.ValidatedProposal,
	policy decision.Policy,
	meta *aiMeta,
) (*planned, error) {
	if validated != nil && validated.IntentProcedureCode != "" {
		if row, ok := byCode[validated.IntentProcedureCode]; ok {
			switch policy.Band(validated.IntentConfidence) {
			case decision.IntentSelect:
				meta.source = "ai"
				var extraction *decision.ExtractionProposal
				if validated.SlotsProcedureCode == row.ProcedureCode && len(validated.Slots) > 0 {
					extraction = toDecisionProposal(validated)
				}
				return decidePinnedWithExtraction(snap, message, row, map[string]decision.SlotValue{}, false, extraction, meta)
			case decision.IntentConfirm:
				meta.source = "ai"
				return confirmIntentFromAI(snap, message, validated, byCode), nil
			}
		}
	}

	// AI abstained, was low-confidence, or named an unknown code: fall back
	// to the exact pre-existing keyword matcher.
	meta.source = "keyword_fallback"
	match := decision.MatchIntentBands(message, candidates)
	switch {
	case match.Band == decision.IntentConfirm:
		return confirmIntent(snap, message, match.Candidates, byCode), nil
	case match.Band == decision.IntentSelect:
		row := byCode[match.Selected.ProcedureCode]
		return decidePinnedWithExtraction(snap, message, row, map[string]decision.SlotValue{}, false, nil, meta)
	default:
		return outOfScope(snap, message, names), nil
	}
}

// decidePinnedWithExtraction mirrors decidePinned but runs
// decision.DecideWithExtraction instead of decision.Decide, and records
// which AI slot keys were accepted/rejected for observability metadata.
// Everything downstream of the decision (guidance, checklist, citations,
// result/save shape) is identical to the keyword-only path because both call
// the same buildPlannedFromDecision.
func decidePinnedWithExtraction(
	snap *repository.TurnSnapshot,
	message string,
	row repository.ProcedureDefinitionRow,
	prior map[string]decision.SlotValue,
	hadPrior bool,
	extraction *decision.ExtractionProposal,
	meta *aiMeta,
) (*planned, error) {
	def, err := decision.ParseDefinition(row.Definition)
	if err != nil {
		return nil, fmt.Errorf("procedure %s: %w", row.ProcedureCode, err)
	}
	decided, applied := decision.DecideWithExtraction(decision.TurnInput{
		Definition:    def,
		Prior:         prior,
		HadPriorState: hadPrior,
		Message:       message,
	}, extraction)
	if meta != nil {
		meta.accepted = append(meta.accepted, applied.Accepted...)
		meta.rejected = append(meta.rejected, applied.Rejected...)
	}
	return buildPlannedFromDecision(snap, message, row, def, decided), nil
}

func toDecisionProposal(validated *aiextract.ValidatedProposal) *decision.ExtractionProposal {
	if validated == nil || len(validated.Slots) == 0 {
		return nil
	}
	out := &decision.ExtractionProposal{Slots: map[string]decision.ExtractedSlot{}}
	for key, s := range validated.Slots {
		out.Slots[key] = decision.ExtractedSlot{Value: s.Value, Operation: s.Operation}
	}
	return out
}

// confirmIntentFromAI turns a validated AI intent (plus its validated
// alternatives) into the same CONFIRM_INTENT flow the keyword engine uses —
// the citizen sees an ordinary "which procedure did you mean?" prompt and
// resolves it via the existing yes/no/name logic in handlePendingConfirm.
// The AI never switches a pinned procedure on its own.
func confirmIntentFromAI(
	snap *repository.TurnSnapshot,
	message string,
	validated *aiextract.ValidatedProposal,
	byCode map[string]repository.ProcedureDefinitionRow,
) *planned {
	seen := map[string]struct{}{}
	cands := make([]PendingCandidate, 0, 1+len(validated.Alternatives))
	add := func(code string, score float64) {
		if code == "" {
			return
		}
		if _, dup := seen[code]; dup {
			return
		}
		row, ok := byCode[code]
		if !ok {
			return
		}
		seen[code] = struct{}{}
		cands = append(cands, PendingCandidate{
			ProcedureCode:      code,
			Name:               row.Name,
			Score:              score,
			ProcedureID:        row.ProcedureID.String(),
			ProcedureVersionID: row.VersionID.String(),
			DefinitionHash:     DefinitionHash(row.Definition),
		})
	}
	add(validated.IntentProcedureCode, validated.IntentConfidence)
	for _, alt := range validated.Alternatives {
		add(alt.ProcedureCode, alt.Confidence)
	}
	return confirmIntentFromPending(snap, message, cands)
}
