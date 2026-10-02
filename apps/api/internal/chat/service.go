package chat

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"
	"unicode"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/aiextract"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// MaxMessageRunes is the shared length limit for every chat write path.
const MaxMessageRunes = 4000

// Service runs one citizen turn inside a database transaction.
//
// When Extractor is nil or AIEnabled is false, Turn runs the original,
// unmodified single-transaction keyword-only path (turnKeywordOnly) — this
// is the exact code that existed before P2 and every Phase 1-3 test exercises
// it unchanged. When AI is enabled, Turn instead runs turnWithAI, a
// three-phase flow (see ai_plan.go) that guarantees the AI network call
// never happens while a database transaction/row lock is held.
type Service struct {
	Pool           *pgxpool.Pool
	Sessions       *repository.SessionRepo
	CitizenDomains []string

	// AI extraction (P2). All four are optional; the zero value fully
	// disables AI and preserves pre-P2 behavior.
	Extractor           aiextract.Extractor
	AIEnabled           bool
	AIPolicy            decision.Policy
	AIPolicySet         bool
	AISlotConfidenceMin float64
	// ClaimLease is how long a PENDING extractor claim stays owned before
	// another request with the same payload may take it over. Zero uses
	// DefaultClaimLease. The lease is not a database lock: it is released
	// before the network call.
	ClaimLease time.Duration
}

// DefaultClaimLease is long enough to cover the hard-capped AI timeout (8s)
// plus the following write transaction, and short enough that a crashed
// owner does not block retries indefinitely.
const DefaultClaimLease = 30 * time.Second

func (s *Service) aiEnabled() bool {
	return s.AIEnabled && s.Extractor != nil
}

// Turn detects or resumes a procedure, decides, and persists the turn.
func (s *Service) Turn(ctx context.Context, sessionID, requestID uuid.UUID, message string) (*TurnResult, error) {
	if s.aiEnabled() {
		return s.turnWithAI(ctx, sessionID, requestID, message)
	}
	return s.turnKeywordOnly(ctx, sessionID, requestID, message)
}

// TurnResult is the decision envelope plus the two persisted messages.
type TurnResult struct {
	SessionID        string                        `json:"session_id"`
	Action           string                        `json:"action"`
	ProcedureCode    string                        `json:"procedure_code,omitempty"`
	ProcedureVersion string                        `json:"procedure_version,omitempty"`
	ReplyText        string                        `json:"reply_text"`
	MissingSlots     []string                      `json:"missing_slots,omitempty"`
	AskNow           []string                      `json:"ask_now,omitempty"`
	Questions        []decision.Question           `json:"questions,omitempty"`
	Guidance         *decision.Guidance            `json:"guidance,omitempty"`
	Citations        []decision.Citation           `json:"citations,omitempty"`
	Candidates       []PendingCandidate            `json:"candidates,omitempty"`
	SlotState        map[string]decision.SlotValue `json:"slot_state,omitempty"`
	FilledSlots      map[string]any                `json:"filled_slots,omitempty"`
	UserMessage      *repository.Message           `json:"user_message,omitempty"`
	AssistantMessage *repository.Message           `json:"assistant_message,omitempty"`
	IdempotentReplay bool                          `json:"idempotent_replay,omitempty"`
}

// PendingCandidate is a confirm option pinned to an exact procedure version.
type PendingCandidate struct {
	ProcedureCode      string  `json:"procedure_code"`
	Name               string  `json:"name"`
	Score              float64 `json:"score"`
	ProcedureID        string  `json:"procedure_id"`
	ProcedureVersionID string  `json:"procedure_version_id"`
	DefinitionHash     string  `json:"definition_hash"`
}

type pendingIntent struct {
	Candidates []PendingCandidate `json:"candidates"`
}

// PayloadHash is the stable hash of a normalized turn payload.
func PayloadHash(message string) string {
	sum := sha256.Sum256([]byte(normalizePayload(message)))
	return hex.EncodeToString(sum[:])
}

func normalizePayload(message string) string {
	trimmed := strings.TrimSpace(message)
	var b strings.Builder
	b.Grow(len(trimmed))
	prevSpace := false
	for _, r := range trimmed {
		if unicode.IsSpace(r) {
			if !prevSpace {
				b.WriteByte(' ')
				prevSpace = true
			}
			continue
		}
		b.WriteRune(r)
		prevSpace = false
	}
	return b.String()
}

// DefinitionHash fingerprints a procedure definition JSON.
func DefinitionHash(def json.RawMessage) string {
	sum := sha256.Sum256(def)
	return hex.EncodeToString(sum[:])
}

// turnKeywordOnly is the pre-P2 single-transaction turn path, byte-for-byte
// unchanged. It is used whenever AI extraction is disabled/unconfigured.
func (s *Service) turnKeywordOnly(ctx context.Context, sessionID, requestID uuid.UUID, message string) (*TurnResult, error) {
	payloadHash := PayloadHash(message)

	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx)

	snap, err := s.Sessions.LockTurn(ctx, tx, sessionID, s.CitizenDomains)
	if err != nil {
		return nil, err
	}
	if snap.Session.Status != "OPEN" {
		return nil, repository.ErrSessionNotOpen
	}

	if prevHash, raw, ok, err := s.Sessions.FindIdempotency(ctx, tx, sessionID, requestID); err != nil {
		return nil, err
	} else if ok {
		if prevHash != payloadHash {
			return nil, repository.ErrIdempotencyConflict
		}
		var replay TurnResult
		if err := json.Unmarshal(raw, &replay); err != nil {
			return nil, err
		}
		replay.IdempotentReplay = true
		if err := tx.Commit(ctx); err != nil {
			return nil, err
		}
		return &replay, nil
	}

	// Legacy: messages exist for (session_id, request_id) but turn_idempotency row is missing.
	if legacy, err := s.replayLegacyTurn(ctx, tx, sessionID, requestID, message, payloadHash); err != nil {
		return nil, err
	} else if legacy != nil {
		if err := tx.Commit(ctx); err != nil {
			return nil, err
		}
		return legacy, nil
	}

	plan, err := planTurn(ctx, s.Sessions, tx, snap, message)
	if err != nil {
		return nil, err
	}
	return s.finishPlan(ctx, tx, sessionID, requestID, snap, message, payloadHash, plan)
}

// finishPlan runs the shared save/idempotency-record/commit tail for a
// *planned turn, used by both the keyword-only path and the AI path
// (ai_plan.go). Identical to the original inline tail of Turn().
func (s *Service) finishPlan(
	ctx context.Context,
	tx pgx.Tx,
	sessionID, requestID uuid.UUID,
	snap *repository.TurnSnapshot,
	message, payloadHash string,
	plan *planned,
) (*TurnResult, error) {
	plan.save.RequestID = requestID
	plan.save.ActorUserID = snap.Session.UserID
	if plan.save.AuditPayload == nil {
		plan.save.AuditPayload = map[string]any{}
	}
	plan.save.AuditPayload["request_id"] = requestID.String()
	plan.save.AuditPayload["session_id"] = sessionID.String()
	plan.save.AuditPayload["xa_id"] = snap.Session.XaID
	plan.save.AuditPayload["payload_hash"] = payloadHash
	if snap.Session.UserID != nil {
		plan.save.AuditPayload["actor_user_id"] = snap.Session.UserID.String()
	} else {
		plan.save.AuditPayload["actor"] = "guest"
	}

	if plan.save.Metadata == nil {
		plan.save.Metadata = map[string]any{}
	}
	plan.save.Metadata["ask_now"] = plan.result.AskNow
	plan.save.Metadata["questions"] = plan.result.Questions
	plan.save.Metadata["guidance"] = plan.result.Guidance
	plan.save.Metadata["slot_state"] = plan.result.SlotState
	plan.save.Metadata["filled_slots"] = plan.result.FilledSlots
	plan.save.Metadata["payload_hash"] = payloadHash
	if len(plan.result.Candidates) > 0 {
		plan.save.Metadata["candidates"] = plan.result.Candidates
	}

	userMsg, asstMsg, err := s.Sessions.SaveTurn(ctx, tx, sessionID, plan.save)
	if err != nil {
		if err == repository.ErrDuplicateTurn {
			if prevHash, raw, ok, findErr := s.Sessions.FindIdempotency(ctx, tx, sessionID, requestID); findErr != nil {
				return nil, findErr
			} else if ok {
				if prevHash != payloadHash {
					return nil, repository.ErrIdempotencyConflict
				}
				var replay TurnResult
				if uErr := json.Unmarshal(raw, &replay); uErr != nil {
					return nil, uErr
				}
				replay.IdempotentReplay = true
				if commitErr := tx.Commit(ctx); commitErr != nil {
					return nil, commitErr
				}
				return &replay, nil
			}
			if legacy, lErr := s.replayLegacyTurn(ctx, tx, sessionID, requestID, message, payloadHash); lErr != nil {
				return nil, lErr
			} else if legacy != nil {
				if commitErr := tx.Commit(ctx); commitErr != nil {
					return nil, commitErr
				}
				return legacy, nil
			}
		}
		return nil, err
	}
	plan.result.UserMessage = userMsg
	plan.result.AssistantMessage = asstMsg

	envelope, err := json.Marshal(plan.result)
	if err != nil {
		return nil, err
	}
	if err := s.Sessions.SaveIdempotency(ctx, tx, sessionID, requestID, payloadHash, envelope); err != nil {
		return nil, err
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return &plan.result, nil
}

// replayLegacyTurn rebuilds a turn envelope from conversation_messages when
// turn_idempotency is absent (pre-000006 rows). Never returns 500 for duplicate retries.
func (s *Service) replayLegacyTurn(
	ctx context.Context,
	tx pgx.Tx,
	sessionID, requestID uuid.UUID,
	message, payloadHash string,
) (*TurnResult, error) {
	userMsg, asstMsg, err := s.Sessions.FindCompletedTurn(ctx, tx, sessionID, requestID)
	if err != nil {
		return nil, err
	}
	if userMsg == nil {
		return nil, nil
	}
	_ = message
	if PayloadHash(userMsg.Content) != payloadHash {
		return nil, repository.ErrIdempotencyConflict
	}
	if asstMsg == nil {
		// Incomplete legacy turn — treat as conflict rather than re-execute.
		return nil, repository.ErrIdempotencyConflict
	}

	meta := asstMsg.Metadata
	if meta == nil {
		meta = map[string]any{}
	}
	action := ""
	if asstMsg.Action != nil {
		action = *asstMsg.Action
	}
	replay := TurnResult{
		SessionID:        sessionID.String(),
		Action:           action,
		ReplyText:        asstMsg.Content,
		UserMessage:      userMsg,
		AssistantMessage: asstMsg,
		IdempotentReplay: true,
		MissingSlots:     stringSliceFromMeta(meta["missing_slots"]),
		AskNow:           stringSliceFromMeta(meta["ask_now"]),
		Questions:        questionsFromMeta(meta["questions"]),
		Guidance:         guidanceFromMeta(meta["guidance"]),
		Citations:        citationsFromMeta(meta["citations"]),
		Candidates:       candidatesFromMeta(meta["candidates"]),
		SlotState:        slotStateFromMeta(meta["slot_state"]),
		FilledSlots:      anyMapFromMeta(meta["filled_slots"]),
		ProcedureCode:    stringFromMeta(meta["procedure_code"]),
		ProcedureVersion: stringFromMeta(meta["procedure_version"]),
	}
	envelope, err := json.Marshal(replay)
	if err != nil {
		return nil, err
	}
	// Best-effort backfill so subsequent retries hit the fast path.
	_ = s.Sessions.SaveIdempotency(ctx, tx, sessionID, requestID, payloadHash, envelope)
	return &replay, nil
}

func stringFromMeta(v any) string {
	if v == nil {
		return ""
	}
	s, _ := v.(string)
	return s
}

func stringSliceFromMeta(v any) []string {
	arr, ok := v.([]any)
	if !ok {
		if s, ok := v.([]string); ok {
			return s
		}
		return nil
	}
	out := make([]string, 0, len(arr))
	for _, item := range arr {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func anyMapFromMeta(v any) map[string]any {
	m, _ := v.(map[string]any)
	return m
}

func questionsFromMeta(v any) []decision.Question {
	raw, err := json.Marshal(v)
	if err != nil || len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var out []decision.Question
	if json.Unmarshal(raw, &out) != nil {
		return nil
	}
	return out
}

func guidanceFromMeta(v any) *decision.Guidance {
	if v == nil {
		return nil
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	var g decision.Guidance
	if json.Unmarshal(raw, &g) != nil {
		return nil
	}
	return &g
}

func citationsFromMeta(v any) []decision.Citation {
	raw, err := json.Marshal(v)
	if err != nil || len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var out []decision.Citation
	if json.Unmarshal(raw, &out) != nil {
		return nil
	}
	return out
}

func candidatesFromMeta(v any) []PendingCandidate {
	raw, err := json.Marshal(v)
	if err != nil || len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var out []PendingCandidate
	if json.Unmarshal(raw, &out) != nil {
		return nil
	}
	return out
}

func slotStateFromMeta(v any) map[string]decision.SlotValue {
	raw, err := json.Marshal(v)
	if err != nil || len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var out map[string]decision.SlotValue
	if json.Unmarshal(raw, &out) != nil {
		return nil
	}
	return out
}

type planned struct {
	result TurnResult
	save   repository.SaveTurnInput
}

func outOfScope(snap *repository.TurnSnapshot, message string, names []string) *planned {
	reply := decision.OutOfScopeReply(names)
	return &planned{
		result: TurnResult{
			SessionID: snap.Session.ID.String(),
			Action:    decision.ActionScope,
			ReplyText: reply,
		},
		save: repository.SaveTurnInput{
			UserContent:        message,
			AssistantContent:   reply,
			Action:             decision.ActionScope,
			Metadata:           map[string]any{},
			ClearPendingIntent: true,
			PinProcedure:       false,
			AuditPayload: map[string]any{
				"action": decision.ActionScope,
			},
		},
	}
}

func planTurn(
	ctx context.Context,
	sessions *repository.SessionRepo,
	tx pgx.Tx,
	snap *repository.TurnSnapshot,
	message string,
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

	if pending, ok := decodePending(snap.PendingIntent); ok && len(pending.Candidates) > 0 {
		handled, err := handlePendingConfirm(ctx, sessions, tx, snap, message, pending, names)
		if err != nil {
			return nil, err
		}
		if handled != nil {
			return handled, nil
		}
	}

	match := decision.MatchIntentBands(message, candidates)

	var row repository.ProcedureDefinitionRow
	var prior map[string]decision.SlotValue
	var hadPrior bool

	switch {
	case match.Band == decision.IntentConfirm && snap.Pinned == nil:
		return confirmIntent(snap, message, match.Candidates, byCode), nil

	case match.Band == decision.IntentSelect && (snap.Pinned == nil || snap.Pinned.ProcedureCode != match.Selected.ProcedureCode):
		row = byCode[match.Selected.ProcedureCode]
		prior = map[string]decision.SlotValue{}
		hadPrior = false

	case snap.Pinned != nil:
		row = *snap.Pinned
		var err error
		prior, err = decodeSlotState(snap.PriorRaw)
		if err != nil {
			return nil, err
		}
		pinnedDef, err := decision.ParseDefinition(row.Definition)
		if err != nil {
			return nil, fmt.Errorf("procedure %s: %w", row.ProcedureCode, err)
		}
		if match.Band != decision.IntentSelect && len(pinnedDef.Missing(prior)) == 0 && !decision.HasSlotCorrection(pinnedDef, prior, message) {
			return outOfScope(snap, message, names), nil
		}
		hadPrior = snap.HadPrior

	case match.Band == decision.IntentConfirm:
		return confirmIntent(snap, message, match.Candidates, byCode), nil

	default:
		return outOfScope(snap, message, names), nil
	}

	return decidePinned(snap, message, row, prior, hadPrior)
}

func handlePendingConfirm(
	ctx context.Context,
	sessions *repository.SessionRepo,
	tx pgx.Tx,
	snap *repository.TurnSnapshot,
	message string,
	pending pendingIntent,
	names []string,
) (*planned, error) {
	folded := decision.Fold(message)

	resolve := func(c PendingCandidate) (*planned, error) {
		procID, err := uuid.Parse(c.ProcedureID)
		if err != nil {
			return outOfScope(snap, message, names), nil
		}
		verID, err := uuid.Parse(c.ProcedureVersionID)
		if err != nil {
			return outOfScope(snap, message, names), nil
		}
		row, err := sessions.LoadProcedureVersion(ctx, tx, procID, verID)
		if err != nil {
			return outOfScope(snap, message, names), nil
		}
		if DefinitionHash(row.Definition) != c.DefinitionHash {
			return outOfScope(snap, message, names), nil
		}
		plannedTurn, err := decidePinned(snap, row.Name, *row, map[string]decision.SlotValue{}, false)
		if err != nil || plannedTurn == nil {
			return outOfScope(snap, message, names), nil
		}
		plannedTurn.save.UserContent = message
		plannedTurn.save.ClearPendingIntent = true
		return plannedTurn, nil
	}

	if isAffirmative(folded) {
		return resolve(pending.Candidates[0])
	}
	if isNegative(folded) {
		reply := "Đã hủy. " + decision.OutOfScopeReply(names)
		return &planned{
			result: TurnResult{
				SessionID: snap.Session.ID.String(),
				Action:    decision.ActionScope,
				ReplyText: reply,
			},
			save: repository.SaveTurnInput{
				UserContent:        message,
				AssistantContent:   reply,
				Action:             decision.ActionScope,
				Metadata:           map[string]any{},
				ClearPendingIntent: true,
				PinProcedure:       false,
				AuditPayload: map[string]any{
					"action":           decision.ActionScope,
					"confirm_response": "no",
					"pending_cleared":  true,
				},
			},
		}, nil
	}
	for _, c := range pending.Candidates {
		if strings.Contains(folded, decision.Fold(c.Name)) || strings.Contains(folded, decision.Fold(c.ProcedureCode)) {
			return resolve(c)
		}
	}
	return confirmIntentFromPending(snap, message, pending.Candidates), nil
}

func confirmIntent(
	snap *repository.TurnSnapshot,
	message string,
	cands []decision.RankedIntent,
	byCode map[string]repository.ProcedureDefinitionRow,
) *planned {
	pending := make([]PendingCandidate, 0, len(cands))
	for _, c := range cands {
		row, ok := byCode[c.ProcedureCode]
		if !ok {
			continue
		}
		pending = append(pending, PendingCandidate{
			ProcedureCode:      c.ProcedureCode,
			Name:               c.Name,
			Score:              c.Score,
			ProcedureID:        row.ProcedureID.String(),
			ProcedureVersionID: row.VersionID.String(),
			DefinitionHash:     DefinitionHash(row.Definition),
		})
	}
	return confirmIntentFromPending(snap, message, pending)
}

func confirmIntentFromPending(snap *repository.TurnSnapshot, message string, cands []PendingCandidate) *planned {
	if len(cands) == 0 {
		return outOfScope(snap, message, nil)
	}
	var b strings.Builder
	b.WriteString("Anh/chị muốn hỏi về thủ tục nào?\n")
	for i, c := range cands {
		fmt.Fprintf(&b, "%d) %s\n", i+1, c.Name)
	}
	b.WriteString("Trả lời Có để chọn mục 1, Không để hủy, hoặc nêu tên thủ tục.")
	reply := strings.TrimSpace(b.String())
	pendingRaw, _ := json.Marshal(pendingIntent{Candidates: cands})
	return &planned{
		result: TurnResult{
			SessionID:  snap.Session.ID.String(),
			Action:     decision.ActionConfirm,
			ReplyText:  reply,
			Candidates: cands,
		},
		save: repository.SaveTurnInput{
			UserContent:      message,
			AssistantContent: reply,
			Action:           decision.ActionConfirm,
			Metadata: map[string]any{
				"candidates": cands,
			},
			PendingIntent: pendingRaw,
			PinProcedure:  false,
			AuditPayload: map[string]any{
				"action":     decision.ActionConfirm,
				"candidates": cands,
			},
		},
	}
}

func decidePinned(
	snap *repository.TurnSnapshot,
	message string,
	row repository.ProcedureDefinitionRow,
	prior map[string]decision.SlotValue,
	hadPrior bool,
) (*planned, error) {
	def, err := decision.ParseDefinition(row.Definition)
	if err != nil {
		return nil, fmt.Errorf("procedure %s: %w", row.ProcedureCode, err)
	}
	decided := decision.Decide(decision.TurnInput{
		Definition:    def,
		Prior:         prior,
		HadPriorState: hadPrior,
		Message:       message,
	})
	return buildPlannedFromDecision(snap, message, row, def, decided), nil
}

// buildPlannedFromDecision turns an already-computed decision.TurnOutput into
// a *planned turn. Shared by the keyword-only path (decidePinned) and the
// AI-assisted path (decidePinnedWithExtraction in ai_plan.go) so both produce
// an identical result shape from a decision.TurnOutput — the LLM never
// touches guidance, checklist, or citations; those always come straight from
// `def`/`decided`, exactly as before P2.
func buildPlannedFromDecision(
	snap *repository.TurnSnapshot,
	message string,
	row repository.ProcedureDefinitionRow,
	def decision.Definition,
	decided decision.TurnOutput,
) *planned {
	procID := row.ProcedureID
	verID := row.VersionID
	meta := map[string]any{
		"procedure_code":    row.ProcedureCode,
		"procedure_version": row.Version,
		"missing_slots":     decided.MissingSlots,
	}
	citations := make([]map[string]any, 0, len(def.Citations))
	for _, c := range def.Citations {
		citations = append(citations, map[string]any{
			"doc_id":         c.DocID,
			"title":          c.Title,
			"source_type":    c.SourceType,
			"effective_date": c.EffectiveDate,
			"issuer":         c.Issuer,
			"page_range":     c.PageHint,
		})
	}
	replyCitations := decided.Citations
	if len(replyCitations) == 0 {
		replyCitations = def.Citations
	}
	if len(replyCitations) > 0 {
		meta["citations"] = replyCitations
	}
	slotState := map[string]any{}
	for key, value := range decided.SlotState {
		slotState[key] = value
	}
	return &planned{
		result: TurnResult{
			SessionID:        snap.Session.ID.String(),
			Action:           decided.Action,
			ProcedureCode:    row.ProcedureCode,
			ProcedureVersion: row.Version,
			ReplyText:        decided.ReplyText,
			MissingSlots:     decided.MissingSlots,
			AskNow:           decided.AskNow,
			Questions:        decided.Questions,
			Guidance:         decided.Guidance,
			Citations:        replyCitations,
			SlotState:        decided.SlotState,
			FilledSlots:      decided.FilledSlots,
		},
		save: repository.SaveTurnInput{
			UserContent:        message,
			AssistantContent:   decided.ReplyText,
			Action:             decided.Action,
			Metadata:           meta,
			ProcedureID:        &procID,
			VersionID:          &verID,
			SlotState:          slotState,
			Citations:          citations,
			ClearPendingIntent: true,
			PinProcedure:       true,
			AuditPayload: map[string]any{
				"action":            decided.Action,
				"procedure_code":    row.ProcedureCode,
				"procedure_version": row.Version,
				"procedure_id":      row.ProcedureID.String(),
				"version_id":        row.VersionID.String(),
				"missing_slots":     decided.MissingSlots,
			},
		},
	}
}

func decodePending(raw json.RawMessage) (pendingIntent, bool) {
	if len(raw) == 0 || string(raw) == "null" || string(raw) == "{}" {
		return pendingIntent{}, false
	}
	var p pendingIntent
	if err := json.Unmarshal(raw, &p); err != nil || len(p.Candidates) == 0 {
		return pendingIntent{}, false
	}
	return p, true
}

func isAffirmative(folded string) bool {
	switch folded {
	case "co", "yes", "dung", "dung roi", "ok", "oke", "uh", "u", "phai", "dong y", "1":
		return true
	}
	return false
}

func isNegative(folded string) bool {
	switch folded {
	case "khong", "ko", "no", "sai", "huy", "khong phai", "0":
		return true
	}
	return false
}

func decodeSlotState(raw []byte) (map[string]decision.SlotValue, error) {
	out := map[string]decision.SlotValue{}
	if len(raw) == 0 || string(raw) == "null" {
		return out, nil
	}
	var wire map[string]struct {
		Value  json.RawMessage `json:"value"`
		Status string          `json:"status"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, fmt.Errorf("decode slot_state: %w", err)
	}
	for key, item := range wire {
		var value any
		if len(item.Value) > 0 && string(item.Value) != "null" {
			if err := json.Unmarshal(item.Value, &value); err != nil {
				return nil, fmt.Errorf("decode slot %s: %w", key, err)
			}
		}
		status := item.Status
		if status == "" {
			status = decision.StatusMissing
		}
		out[key] = decision.SlotValue{Value: value, Status: status}
	}
	return out, nil
}
