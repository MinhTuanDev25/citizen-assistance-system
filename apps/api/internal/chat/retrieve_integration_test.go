//go:build integration

package chat_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestHTTPPinnedSourcesSurviveReindexUnlinkAndV2(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()
	eng := testEngine(t, pool)
	ctx := context.Background()

	admin := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	procID := uuid.New()
	verID := uuid.New()
	procCode := "p5_pin_" + uuid.NewString()[:8]
	token := "phase5pin" + uuid.NewString()
	def := fmt.Sprintf(`{
		"procedure_code": %q,
		"name": %q,
		"intent_examples": [%q],
		"slots": {},
		"required_slots": [],
		"guidance": {"summary": "Hướng dẫn chứng thực", "checklist": ["Bản chính giấy tờ cần chứng thực"], "where_to_submit": "UBND xã"},
		"citations": []
	}`, procCode, procCode, token)
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedures (id, procedure_code, domain_id, name, xa_id)
		VALUES ($1,$2,'ho_tich_chung_thuc',$2,'xa_chu_se')`, procID, procCode); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedure_versions (id, procedure_id, version, status, definition, created_by)
		VALUES ($1,$2,'1.0.0','ACTIVE',$3::jsonb,$4)`, verID, procID, def, admin); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE procedures SET active_version_id=$2 WHERE id=$1`, procID, verID); err != nil {
		t.Fatal(err)
	}

	g1Excerpt := token + " Bản chính giấy tờ cần chứng thực khi nộp tại UBND xã Chư Sê."
	g2Excerpt := "Văn bản thế hệ hai không thuộc session đã ghim."
	docID, g1, g2 := insertTwoGenerations(t, pool, procID, verID, g1Excerpt, g2Excerpt)
	t.Cleanup(func() {
		cleanupIndexedDoc(pool, docID)
		_, _ = pool.Exec(ctx, `DELETE FROM session_source_pin_sets WHERE procedure_id=$1`, procID)
		_, _ = pool.Exec(ctx, `UPDATE conversation_sessions SET active_procedure_id=NULL, active_procedure_version_id=NULL WHERE active_procedure_id=$1`, procID)
		_, _ = pool.Exec(ctx, `DELETE FROM session_slot_states WHERE procedure_id=$1`, procID)
		_, _ = pool.Exec(ctx, `UPDATE procedures SET active_version_id=NULL WHERE id=$1`, procID)
		_, _ = pool.Exec(ctx, `DELETE FROM procedure_versions WHERE procedure_id=$1`, procID)
		_, _ = pool.Exec(ctx, `DELETE FROM procedures WHERE id=$1`, procID)
	})

	_, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions", map[string]any{"xa_id": "xa_chu_se"}, nil)
	d := dataMap(env)
	sessionID, _ := d["id"].(string)
	guest, _ := d["guest_token"].(string)
	if sessionID == "" {
		t.Fatalf("session %v", env)
	}
	code, turn := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": token},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("turn %d %v", code, turn)
	}
	td := dataMap(turn)
	if fmt.Sprint(td["action"]) != "DIRECT_ANSWER" && fmt.Sprint(td["action"]) != "PROVIDE_FINAL_GUIDANCE" {
		t.Fatalf("want guidance action, got %v", td)
	}
	assertIndexedCitation(t, td, g1Excerpt, g1)
	if fmt.Sprint(td["retrieval_status"]) != "ok" {
		t.Fatalf("retrieval_status %v", td["retrieval_status"])
	}

	if _, err := pool.Exec(ctx, `UPDATE procedure_version_documents SET active_generation_id=$2 WHERE document_id=$1`, docID, g2); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE document_index_generations SET status='SUPERSEDED' WHERE id=$1`, g1); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE document_index_generations SET status='READY' WHERE id=$1`, g2); err != nil {
		t.Fatal(err)
	}

	code, turn2 := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": token + " lần hai"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("turn2 %d %v", code, turn2)
	}
	td2 := dataMap(turn2)
	assertIndexedCitation(t, td2, g1Excerpt, g1)
	if citationHasExcerpt(td2, g2Excerpt) {
		t.Fatal("S1 retrieved G2 after reindex")
	}

	if _, err := pool.Exec(ctx, `
		UPDATE procedure_version_documents
		SET unlinked_at = now(), active_generation_id = NULL
		WHERE document_id=$1 AND procedure_version_id=$2`, docID, verID); err != nil {
		t.Fatal(err)
	}
	code, hist := apiJSON(t, eng, http.MethodGet, "/api/v1/sessions/"+sessionID+"/messages?limit=20", nil,
		map[string]string{"X-Guest-Token": guest})
	if code != http.StatusOK {
		t.Fatalf("history %d %v", code, hist)
	}
	items, _ := dataMap(hist)["items"].([]any)
	found := false
	for _, raw := range items {
		item, _ := raw.(map[string]any)
		meta, _ := item["metadata"].(map[string]any)
		cites, _ := meta["citations"].([]any)
		for _, c := range cites {
			row, _ := c.(map[string]any)
			if fmt.Sprint(row["excerpt"]) == g1Excerpt || fmt.Sprint(row["Excerpt"]) == g1Excerpt {
				found = true
			}
			if fmt.Sprint(row["excerpt"]) == g2Excerpt {
				t.Fatal("history remapped to G2 excerpt")
			}
		}
	}
	if !found {
		t.Fatalf("historical excerpt missing: %v", hist)
	}

	var pinGen uuid.UUID
	if err := pool.QueryRow(ctx, `
		SELECT generation_id FROM session_source_pins WHERE session_id=$1`, sessionID).Scan(&pinGen); err != nil {
		t.Fatal(err)
	}
	if pinGen != g1 {
		t.Fatalf("pin %s want G1 %s", pinGen, g1)
	}
}

func TestAskTurnPinsSourcesBeforeReindexAndV2(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()
	eng := testEngine(t, pool)
	ctx := context.Background()

	admin := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	procID := uuid.New()
	verID := uuid.New()
	v2ID := uuid.New()
	procCode := "p5_ask_" + uuid.NewString()[:8]
	token := "phase5ask" + uuid.NewString()
	def := fmt.Sprintf(`{
		"procedure_code": %q,
		"name": %q,
		"intent_examples": [%q],
		"slots": {
			"noi_sinh": {"type": "string", "question": "Bé sinh ở đâu?"},
			"da_ket_hon": {"type": "boolean", "question": "Cha mẹ bé đã đăng ký kết hôn chưa?"}
		},
		"required_slots": ["noi_sinh", "da_ket_hon"],
		"guidance": {"summary": "Hướng dẫn khai sinh", "checklist": ["Bản chính giấy chứng sinh"], "where_to_submit": "UBND xã"},
		"citations": []
	}`, procCode, procCode, token)
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedures (id, procedure_code, domain_id, name, xa_id)
		VALUES ($1,$2,'ho_tich_chung_thuc',$2,'xa_chu_se')`, procID, procCode); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedure_versions (id, procedure_id, version, status, definition, created_by)
		VALUES ($1,$2,'1.0.0','ACTIVE',$3::jsonb,$4)`, verID, procID, def, admin); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE procedures SET active_version_id=$2 WHERE id=$1`, procID, verID); err != nil {
		t.Fatal(err)
	}

	g1Excerpt := "Đăng ký khai sinh tại Bệnh viện huyện. Bản chính giấy chứng sinh."
	g2Excerpt := "Văn bản thế hệ hai không thuộc session đang hỏi slot."
	docID, g1, g2 := insertTwoGenerations(t, pool, procID, verID, g1Excerpt, g2Excerpt)
	t.Cleanup(func() {
		cleanupIndexedDoc(pool, docID)
		_, _ = pool.Exec(ctx, `DELETE FROM session_source_pin_sets WHERE procedure_id=$1`, procID)
		_, _ = pool.Exec(ctx, `UPDATE conversation_sessions SET active_procedure_id=NULL, active_procedure_version_id=NULL WHERE active_procedure_id=$1`, procID)
		_, _ = pool.Exec(ctx, `DELETE FROM session_slot_states WHERE procedure_id=$1`, procID)
		_, _ = pool.Exec(ctx, `UPDATE procedures SET active_version_id=NULL WHERE id=$1`, procID)
		_, _ = pool.Exec(ctx, `DELETE FROM procedure_versions WHERE procedure_id=$1`, procID)
		_, _ = pool.Exec(ctx, `DELETE FROM procedures WHERE id=$1`, procID)
	})

	_, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions", map[string]any{"xa_id": "xa_chu_se"}, nil)
	d := dataMap(env)
	sessionID, _ := d["id"].(string)
	guest, _ := d["guest_token"].(string)
	code, turn := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": token},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("ask turn %d %v", code, turn)
	}
	td := dataMap(turn)
	if fmt.Sprint(td["action"]) != "ASK_MISSING_SLOTS" {
		t.Fatalf("want ASK_MISSING_SLOTS, got %v procedure=%v missing=%v", td["action"], td["procedure_code"], td["missing_slots"])
	}
	if cites, _ := td["citations"].([]any); len(cites) != 0 {
		t.Fatalf("ask turn must not attach PDF appendix, got %v", cites)
	}
	var pinGen, pinnedVer uuid.UUID
	if err := pool.QueryRow(ctx, `
		SELECT generation_id FROM session_source_pins WHERE session_id=$1`, sessionID).Scan(&pinGen); err != nil {
		t.Fatalf("sources must freeze on ASK: %v", err)
	}
	if pinGen != g1 {
		t.Fatalf("ask pin %s want G1 %s", pinGen, g1)
	}
	if err := pool.QueryRow(ctx, `
		SELECT active_procedure_version_id FROM conversation_sessions WHERE id=$1`, sessionID).Scan(&pinnedVer); err != nil {
		t.Fatal(err)
	}
	if pinnedVer != verID {
		t.Fatalf("session version %s want V1 %s", pinnedVer, verID)
	}

	if _, err := pool.Exec(ctx, `UPDATE procedure_version_documents SET active_generation_id=$2 WHERE document_id=$1`, docID, g2); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE document_index_generations SET status='SUPERSEDED' WHERE id=$1`, g1); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE document_index_generations SET status='READY' WHERE id=$1`, g2); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE procedure_versions SET status='ARCHIVED' WHERE id=$1`, verID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedure_versions (id, procedure_id, version, status, definition, created_by)
		VALUES ($1,$2,'2.0.0','ACTIVE',$3::jsonb,$4)`, v2ID, procID, def, admin); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE procedures SET active_version_id=$2 WHERE id=$1`, procID, v2ID); err != nil {
		t.Fatal(err)
	}

	code, turn2 := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "Bệnh viện huyện Chư Sê"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("follow-up turn %d %v", code, turn2)
	}
	td2 := dataMap(turn2)
	if fmt.Sprint(td2["action"]) != "ASK_MISSING_SLOTS" {
		t.Fatalf("session still missing slots after V2/reindex, got %v procedure=%v missing=%v", td2["action"], td2["procedure_code"], td2["missing_slots"])
	}
	if citationHasExcerpt(td2, g2Excerpt) {
		t.Fatal("session asked slots on V1 but retrieved G2 after reindex/V2")
	}
	if err := pool.QueryRow(ctx, `
		SELECT generation_id FROM session_source_pins WHERE session_id=$1`, sessionID).Scan(&pinGen); err != nil {
		t.Fatal(err)
	}
	if pinGen != g1 {
		t.Fatalf("pin moved to %s after V2/reindex", pinGen)
	}
	if err := pool.QueryRow(ctx, `
		SELECT active_procedure_version_id FROM conversation_sessions WHERE id=$1`, sessionID).Scan(&pinnedVer); err != nil {
		t.Fatal(err)
	}
	if pinnedVer != verID {
		t.Fatalf("session jumped to %s after V2 activate", pinnedVer)
	}

	code, turn3 := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "rồi, khai sinh tại Bệnh viện huyện"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("final turn %d %v", code, turn3)
	}
	td3 := dataMap(turn3)
	if fmt.Sprint(td3["action"]) != "DIRECT_ANSWER" && fmt.Sprint(td3["action"]) != "PROVIDE_FINAL_GUIDANCE" {
		t.Fatalf("want final guidance after filling slots, got %v missing=%v", td3["action"], td3["missing_slots"])
	}
	if fmt.Sprint(td3["procedure_version"]) != "1.0.0" {
		t.Fatalf("checklist version jumped: %v", td3["procedure_version"])
	}
	assertIndexedCitation(t, td3, g1Excerpt, g1)
	if citationHasExcerpt(td3, g2Excerpt) {
		t.Fatal("final guidance retrieved G2 after V2/reindex")
	}

	code, hist := apiJSON(t, eng, http.MethodGet, "/api/v1/sessions/"+sessionID+"/messages?limit=20", nil,
		map[string]string{"X-Guest-Token": guest})
	if code != http.StatusOK {
		t.Fatalf("history %d %v", code, hist)
	}
	if !historyHasExcerpt(dataMap(hist), g1Excerpt) {
		t.Fatal("historical excerpt missing after V2")
	}
	if historyHasExcerpt(dataMap(hist), g2Excerpt) {
		t.Fatal("history remapped to G2")
	}

	_, env2 := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions", map[string]any{"xa_id": "xa_chu_se"}, nil)
	s2 := dataMap(env2)
	s2ID, _ := s2["id"].(string)
	s2Guest, _ := s2["guest_token"].(string)
	code, s2turn := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+s2ID+"/turns",
		map[string]any{"message": token},
		map[string]string{"X-Guest-Token": s2Guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("S2 %d %v", code, s2turn)
	}
	if fmt.Sprint(dataMap(s2turn)["procedure_version"]) == "1.0.0" && fmt.Sprint(dataMap(s2turn)["action"]) == "ASK_MISSING_SLOTS" {
		t.Fatal("new session S2 should follow active V2, not stay on archived 1.0.0 after a matching intent")
	}
}

func TestConcurrentActivateDoesNotMixPinnedVersionAndSources(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()
	eng := testEngine(t, pool)
	ctx := context.Background()
	admin := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	procID := uuid.New()
	verID := uuid.New()
	v2ID := uuid.New()
	procCode := "p5_race_" + uuid.NewString()[:8]
	token := "phase5race" + uuid.NewString()
	def := fmt.Sprintf(`{
		"procedure_code": %q,
		"name": %q,
		"intent_examples": [%q],
		"slots": {},
		"required_slots": [],
		"guidance": {"summary": "Hướng dẫn chứng thực", "checklist": ["Bản chính giấy tờ cần chứng thực"], "where_to_submit": "UBND xã"},
		"citations": []
	}`, procCode, procCode, token)
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedures (id, procedure_code, domain_id, name, xa_id)
		VALUES ($1,$2,'ho_tich_chung_thuc',$2,'xa_chu_se')`, procID, procCode); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedure_versions (id, procedure_id, version, status, definition, created_by)
		VALUES ($1,$2,'1.0.0','ACTIVE',$3::jsonb,$4)`, verID, procID, def, admin); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE procedures SET active_version_id=$2 WHERE id=$1`, procID, verID); err != nil {
		t.Fatal(err)
	}
	g1Excerpt := token + " Bản chính giấy tờ cần chứng thực khi nộp tại UBND xã Chư Sê."
	g2Excerpt := "Thế hệ hai không thuộc session đang chat."
	docID, g1, g2 := insertTwoGenerations(t, pool, procID, verID, g1Excerpt, g2Excerpt)
	t.Cleanup(func() {
		cleanupIndexedDoc(pool, docID)
		_, _ = pool.Exec(ctx, `DELETE FROM session_source_pin_sets WHERE procedure_id=$1`, procID)
		_, _ = pool.Exec(ctx, `UPDATE conversation_sessions SET active_procedure_id=NULL, active_procedure_version_id=NULL WHERE active_procedure_id=$1`, procID)
		_, _ = pool.Exec(ctx, `DELETE FROM session_slot_states WHERE procedure_id=$1`, procID)
		_, _ = pool.Exec(ctx, `UPDATE procedures SET active_version_id=NULL WHERE id=$1`, procID)
		_, _ = pool.Exec(ctx, `DELETE FROM procedure_versions WHERE procedure_id=$1`, procID)
		_, _ = pool.Exec(ctx, `DELETE FROM procedures WHERE id=$1`, procID)
	})

	_, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions", map[string]any{"xa_id": "xa_chu_se"}, nil)
	d := dataMap(env)
	sessionID, _ := d["id"].(string)
	guest, _ := d["guest_token"].(string)
	code, turn := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": token},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("pin turn %d %v", code, turn)
	}

	start := make(chan struct{})
	errCh := make(chan error, 2)
	go func() {
		<-start
		_, err := pool.Exec(ctx, `UPDATE procedure_version_documents SET active_generation_id=$2 WHERE document_id=$1`, docID, g2)
		if err != nil {
			errCh <- err
			return
		}
		_, err = pool.Exec(ctx, `UPDATE document_index_generations SET status='SUPERSEDED' WHERE id=$1`, g1)
		if err != nil {
			errCh <- err
			return
		}
		_, err = pool.Exec(ctx, `UPDATE document_index_generations SET status='READY' WHERE id=$1`, g2)
		if err != nil {
			errCh <- err
			return
		}
		_, err = pool.Exec(ctx, `UPDATE procedure_versions SET status='ARCHIVED' WHERE id=$1`, verID)
		if err != nil {
			errCh <- err
			return
		}
		_, err = pool.Exec(ctx, `
			INSERT INTO procedure_versions (id, procedure_id, version, status, definition, created_by)
			VALUES ($1,$2,'2.0.0','ACTIVE',$3::jsonb,$4)`, v2ID, procID, def, admin)
		if err != nil {
			errCh <- err
			return
		}
		_, err = pool.Exec(ctx, `UPDATE procedures SET active_version_id=$2 WHERE id=$1`, procID, v2ID)
		errCh <- err
	}()
	go func() {
		<-start
		code, turn2 := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
			map[string]any{"message": token + " lần hai"},
			map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
		if code != http.StatusOK {
			errCh <- fmt.Errorf("turn2 %d %v", code, turn2)
			return
		}
		td2 := dataMap(turn2)
		if citationHasExcerpt(td2, g2Excerpt) {
			errCh <- fmt.Errorf("concurrent activate leaked G2")
			return
		}
		errCh <- nil
	}()
	close(start)
	if err := <-errCh; err != nil {
		t.Fatal(err)
	}
	if err := <-errCh; err != nil {
		t.Fatal(err)
	}

	var pinGen, pinnedVer uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT generation_id FROM session_source_pins WHERE session_id=$1`, sessionID).Scan(&pinGen); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT active_procedure_version_id FROM conversation_sessions WHERE id=$1`, sessionID).Scan(&pinnedVer); err != nil {
		t.Fatal(err)
	}
	if pinGen != g1 || pinnedVer != verID {
		t.Fatalf("mixed state pin=%s want %s version=%s want %s", pinGen, g1, pinnedVer, verID)
	}
}

type pinActivateFix struct {
	procID, verID, v2ID, g1, v2Gen uuid.UUID
	token, adminTok, activatePath  string
}

func setupPinActivateFixture(t *testing.T, pool *pgxpool.Pool) pinActivateFix {
	t.Helper()
	ctx := context.Background()
	admin := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	fx := pinActivateFix{
		procID: uuid.New(),
		verID:  uuid.New(),
		v2ID:   uuid.New(),
		token:  "phase5firstpin" + uuid.NewString(),
	}
	procCode := "p5_firstpin_" + uuid.NewString()[:8]
	defV1 := fmt.Sprintf(`{
		"procedure_code": %q,
		"name": %q,
		"version": "1.0.0",
		"xa_id": "xa_chu_se",
		"domain": "ho_tich_chung_thuc",
		"status": "APPROVED",
		"intent_examples": [%q],
		"slots": {},
		"required_slots": [],
		"guidance": {"summary": "Hướng dẫn chứng thực", "checklist": ["Bản chính giấy tờ cần chứng thực"], "where_to_submit": "UBND xã"},
		"citations": []
	}`, procCode, procCode, fx.token)
	defV2 := fmt.Sprintf(`{
		"procedure_code": %q,
		"name": %q,
		"version": "2.0.0",
		"xa_id": "xa_chu_se",
		"domain": "ho_tich_chung_thuc",
		"status": "APPROVED",
		"intent_examples": [%q],
		"slots": {},
		"required_slots": [],
		"guidance": {"summary": "Hướng dẫn chứng thực V2", "checklist": ["Bản chính giấy tờ cần chứng thực"], "where_to_submit": "UBND xã"},
		"citations": []
	}`, procCode, procCode, fx.token)
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedures (id, procedure_code, domain_id, name, xa_id)
		VALUES ($1,$2,'ho_tich_chung_thuc',$2,'xa_chu_se')`, fx.procID, procCode); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedure_versions (id, procedure_id, version, status, definition, created_by)
		VALUES ($1,$2,'1.0.0','ACTIVE',$3::jsonb,$4)`, fx.verID, fx.procID, defV1, admin); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE procedures SET active_version_id=$2 WHERE id=$1`, fx.procID, fx.verID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedure_versions (id, procedure_id, version, status, definition, created_by, approved_by, approved_at)
		VALUES ($1,$2,'2.0.0','APPROVED',$3::jsonb,$4,$4,now())`, fx.v2ID, fx.procID, defV2, admin); err != nil {
		t.Fatal(err)
	}
	g1Excerpt := fx.token + " Bản chính giấy tờ cần chứng thực khi nộp tại UBND xã Chư Sê."
	v2Excerpt := fx.token + " Nguồn V2 không được trộn vào pin V1."
	docID, g1, _ := insertTwoGenerations(t, pool, fx.procID, fx.verID, g1Excerpt, "Thế hệ hai không thuộc session đang chat.")
	v2Doc, v2Gen := insertApprovedReadySource(t, pool, fx.procID, fx.v2ID, v2Excerpt)
	fx.g1, fx.v2Gen = g1, v2Gen
	t.Cleanup(func() {
		cleanupIndexedDoc(pool, docID)
		cleanupIndexedDoc(pool, v2Doc)
		_, _ = pool.Exec(ctx, `DELETE FROM session_source_pins WHERE session_id IN (SELECT id FROM conversation_sessions WHERE active_procedure_id=$1)`, fx.procID)
		_, _ = pool.Exec(ctx, `DELETE FROM session_source_pin_sets WHERE procedure_id=$1`, fx.procID)
		_, _ = pool.Exec(ctx, `UPDATE conversation_sessions SET active_procedure_id=NULL, active_procedure_version_id=NULL WHERE active_procedure_id=$1`, fx.procID)
		_, _ = pool.Exec(ctx, `DELETE FROM session_slot_states WHERE procedure_id=$1`, fx.procID)
		_, _ = pool.Exec(ctx, `DELETE FROM audit_logs WHERE entity_id=$1 OR payload->>'procedure_id'=$1`, fx.procID.String())
		_, _ = pool.Exec(ctx, `DELETE FROM admin_write_idempotency WHERE action='ACTIVATE_VERSION' AND response_json->>'procedure_id'=$1`, fx.procID.String())
		_, _ = pool.Exec(ctx, `UPDATE procedures SET active_version_id=NULL WHERE id=$1`, fx.procID)
		_, _ = pool.Exec(ctx, `DELETE FROM procedure_versions WHERE procedure_id=$1`, fx.procID)
		_, _ = pool.Exec(ctx, `DELETE FROM procedures WHERE id=$1`, fx.procID)
	})
	tokens := &auth.TokenService{Secret: []byte("integration-test-jwt-secret-key"), TTL: time.Hour}
	adminTok, _, err := tokens.Issue(admin, auth.RoleAdmin, "admin@chuse.vn", "Seed Admin")
	if err != nil {
		t.Fatal(err)
	}
	fx.adminTok = adminTok
	fx.activatePath = fmt.Sprintf("/api/v1/admin/procedures/%s/versions/%s/activate", fx.procID, fx.v2ID)
	return fx
}

func adminActivateV2(t *testing.T, eng http.Handler, fx pinActivateFix) {
	t.Helper()
	code, env := apiJSON(t, eng, http.MethodPost, fx.activatePath,
		map[string]any{"expected_active_version_id": fx.verID.String()},
		map[string]string{"Authorization": "Bearer " + fx.adminTok, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("admin activate %d %v", code, env)
	}
}

func firstPinTurn(t *testing.T, eng http.Handler, sessionID, guest, token string) {
	t.Helper()
	code, turn := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": token},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("first pin turn %d %v", code, turn)
	}
}

func openGuestSession(t *testing.T, eng http.Handler) (sessionID, guest string) {
	t.Helper()
	_, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions", map[string]any{"xa_id": "xa_chu_se"}, nil)
	d := dataMap(env)
	sessionID, _ = d["id"].(string)
	guest, _ = d["guest_token"].(string)
	if sessionID == "" || guest == "" {
		t.Fatalf("session env=%v", env)
	}
	return sessionID, guest
}

func assertLiveVersion(t *testing.T, pool *pgxpool.Pool, procID, want uuid.UUID) {
	t.Helper()
	var live uuid.UUID
	if err := pool.QueryRow(context.Background(), `SELECT active_version_id FROM procedures WHERE id=$1`, procID).Scan(&live); err != nil {
		t.Fatal(err)
	}
	if live != want {
		t.Fatalf("live version %s want %s", live, want)
	}
}

func assertSessionPinsAligned(t *testing.T, pool *pgxpool.Pool, sessionID string, fx pinActivateFix, wantVer uuid.UUID) {
	t.Helper()
	var sessionVer, pinSetVer uuid.UUID
	if err := pool.QueryRow(context.Background(), `
		SELECT s.active_procedure_version_id, ps.procedure_version_id
		FROM conversation_sessions s
		JOIN session_source_pin_sets ps ON ps.session_id = s.id
		WHERE s.id = $1`, sessionID).Scan(&sessionVer, &pinSetVer); err != nil {
		t.Fatal(err)
	}
	if sessionVer != pinSetVer {
		t.Fatalf("mixed-version state session=%s pin_set=%s", sessionVer, pinSetVer)
	}
	if sessionVer != wantVer {
		t.Fatalf("session version %s want %s", sessionVer, wantVer)
	}
	rows, err := pool.Query(context.Background(), `
		SELECT procedure_version_id, generation_id FROM session_source_pins WHERE session_id=$1`, sessionID)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	pinCount := 0
	for rows.Next() {
		var pinVer, pinGen uuid.UUID
		if err := rows.Scan(&pinVer, &pinGen); err != nil {
			t.Fatal(err)
		}
		pinCount++
		if pinVer != sessionVer {
			t.Fatalf("pin version %s != session %s gen=%s", pinVer, sessionVer, pinGen)
		}
		switch sessionVer {
		case fx.verID:
			if pinGen != fx.g1 {
				t.Fatalf("V1 session pinned generation %s want G1 %s", pinGen, fx.g1)
			}
		case fx.v2ID:
			if pinGen != fx.v2Gen {
				t.Fatalf("V2 session pinned generation %s want V2 gen %s", pinGen, fx.v2Gen)
			}
		default:
			t.Fatalf("session pinned unknown version %s", sessionVer)
		}
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if pinCount < 1 {
		t.Fatal("first pin must freeze at least one READY source")
	}
}

func TestFirstPinThenAdminActivateKeepsV1Pins(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()
	eng := testEngine(t, pool)
	fx := setupPinActivateFixture(t, pool)
	sessionID, guest := openGuestSession(t, eng)
	firstPinTurn(t, eng, sessionID, guest, fx.token)
	adminActivateV2(t, eng, fx)
	assertLiveVersion(t, pool, fx.procID, fx.v2ID)
	assertSessionPinsAligned(t, pool, sessionID, fx, fx.verID)
}

func TestAdminActivateThenFirstPinUsesV2(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()
	eng := testEngine(t, pool)
	fx := setupPinActivateFixture(t, pool)
	sessionID, guest := openGuestSession(t, eng)
	adminActivateV2(t, eng, fx)
	firstPinTurn(t, eng, sessionID, guest, fx.token)
	assertLiveVersion(t, pool, fx.procID, fx.v2ID)
	assertSessionPinsAligned(t, pool, sessionID, fx, fx.v2ID)
}

func TestConcurrentActivateDuringFirstPinKeepsVersionAligned(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()
	eng := testEngine(t, pool)
	fx := setupPinActivateFixture(t, pool)
	sessionID, guest := openGuestSession(t, eng)

	start := make(chan struct{})
	errCh := make(chan error, 2)
	go func() {
		<-start
		code, turn := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
			map[string]any{"message": fx.token},
			map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
		if code != http.StatusOK {
			errCh <- fmt.Errorf("first pin turn %d %v", code, turn)
			return
		}
		errCh <- nil
	}()
	go func() {
		<-start
		code, activateEnv := apiJSON(t, eng, http.MethodPost, fx.activatePath,
			map[string]any{"expected_active_version_id": fx.verID.String()},
			map[string]string{"Authorization": "Bearer " + fx.adminTok, "X-Request-ID": uuid.NewString()})
		if code != http.StatusOK {
			errCh <- fmt.Errorf("admin activate %d %v", code, activateEnv)
			return
		}
		errCh <- nil
	}()
	close(start)
	if err := <-errCh; err != nil {
		t.Fatal(err)
	}
	if err := <-errCh; err != nil {
		t.Fatal(err)
	}
	assertLiveVersion(t, pool, fx.procID, fx.v2ID)
	var sessionVer uuid.UUID
	if err := pool.QueryRow(context.Background(), `SELECT active_procedure_version_id FROM conversation_sessions WHERE id=$1`, sessionID).Scan(&sessionVer); err != nil {
		t.Fatal(err)
	}
	assertSessionPinsAligned(t, pool, sessionID, fx, sessionVer)
}

func historyHasExcerpt(hist map[string]any, excerpt string) bool {
	items, _ := hist["items"].([]any)
	for _, raw := range items {
		item, _ := raw.(map[string]any)
		meta, _ := item["metadata"].(map[string]any)
		cites, _ := meta["citations"].([]any)
		for _, c := range cites {
			row, _ := c.(map[string]any)
			if fmt.Sprint(row["excerpt"]) == excerpt || fmt.Sprint(row["Excerpt"]) == excerpt {
				return true
			}
		}
	}
	return false
}

func assertIndexedCitation(t *testing.T, td map[string]any, excerpt string, gen uuid.UUID) {
	t.Helper()
	cites, _ := td["citations"].([]any)
	for _, raw := range cites {
		row, _ := raw.(map[string]any)
		if fmt.Sprint(row["source_type"]) != "indexed_pdf" {
			continue
		}
		if fmt.Sprint(row["excerpt"]) == excerpt && fmt.Sprint(row["generation_id"]) == gen.String() {
			return
		}
	}
	t.Fatalf("missing indexed citation excerpt=%q gen=%s in %v", excerpt, gen, td["citations"])
}

func citationHasExcerpt(td map[string]any, excerpt string) bool {
	cites, _ := td["citations"].([]any)
	for _, raw := range cites {
		row, _ := raw.(map[string]any)
		if fmt.Sprint(row["excerpt"]) == excerpt {
			return true
		}
	}
	return false
}

func insertApprovedReadySource(t *testing.T, pool *pgxpool.Pool, procID, versionID uuid.UUID, excerpt string) (docID, genID uuid.UUID) {
	t.Helper()
	ctx := context.Background()
	admin := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	docID = uuid.New()
	genID = uuid.New()
	jobID := uuid.New()
	sum := func(s string) string {
		h := sha256.Sum256([]byte(s))
		return hex.EncodeToString(h[:])
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO documents (
			id, xa_id, domain_id, title, filename, storage_uri, checksum, mime_type, file_size_bytes,
			processing_status, validity_status, uploaded_by, document_number
		) VALUES (
			$1,'xa_chu_se','ho_tich_chung_thuc','HD V2 PDF','v2.pdf','s3://cas-documents/v2',$2,
			'application/pdf', 12, 'READY', 'VALID', $3, '02/HD-V2'
		)`, docID, sum(docID.String()), admin); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedure_version_documents (
			procedure_version_id, document_id, procedure_id, xa_id, domain_id, relationship_type, index_status
		) VALUES ($1,$2,$3,'xa_chu_se','ho_tich_chung_thuc','SOURCE','UPLOADED')`,
		versionID, docID, procID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO document_index_jobs (
			id, document_id, xa_id, request_id, payload_hash, status, procedure_version_id, claim_token, finished_at
		) VALUES ($1,$2,'xa_chu_se',$3,'v2hash','SUCCEEDED',$4,gen_random_uuid(), now())`,
		jobID, docID, uuid.New(), versionID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO document_index_generations (
			id, xa_id, document_id, procedure_version_id, job_id, status,
			pipeline_version, extraction_version, ocr_version, chunk_config_hash,
			embedding_model_id, embedding_revision, embedding_checksum, vector_dimension,
			source_sha256, content_sha256, manifest_hash, chunk_count, vector_count, published_at
		) VALUES (
			$1,'xa_chu_se',$2,$3,$4,'READY',
			'p5','p5','p5','p5','e5-small','rev','',384,
			$5,$5,'',1,1, now()
		)`, genID, docID, versionID, jobID, sum(genID.String())); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		UPDATE procedure_version_documents
		SET index_status='READY', active_generation_id=$3
		WHERE procedure_version_id=$1 AND document_id=$2`, versionID, docID, genID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO knowledge_chunks (
			id, xa_id, procedure_id, procedure_version_id, document_id, chunk_index, content,
			generation_id, page_start, page_end, text_sha256, token_count, extraction_source
		) VALUES ($1,'xa_chu_se',$2,$3,$4,0,$5,$6,3,3,$7,8,'native')`,
		uuid.New(), procID, versionID, docID, excerpt, genID, sum(excerpt)); err != nil {
		t.Fatal(err)
	}
	return docID, genID
}

func insertTwoGenerations(t *testing.T, pool *pgxpool.Pool, procID, verID uuid.UUID, g1Text, g2Text string) (docID, g1, g2 uuid.UUID) {
	t.Helper()
	ctx := context.Background()
	admin := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	docID = uuid.New()
	g1 = uuid.New()
	g2 = uuid.New()
	job1 := uuid.New()
	job2 := uuid.New()
	sum := func(s string) string {
		h := sha256.Sum256([]byte(s))
		return hex.EncodeToString(h[:])
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO documents (
			id, xa_id, domain_id, title, filename, storage_uri, checksum, mime_type, file_size_bytes,
			processing_status, validity_status, uploaded_by, document_number
		) VALUES (
			$1,'xa_chu_se','ho_tich_chung_thuc','HD chứng thực PDF','ct.pdf','s3://cas-documents/ct',$2,
			'application/pdf', 12, 'READY', 'VALID', $3, '01/HD-CT'
		)`, docID, sum(docID.String()), admin); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedure_version_documents (
			procedure_version_id, document_id, procedure_id, xa_id, domain_id, relationship_type, index_status
		) VALUES ($1,$2,$3,'xa_chu_se','ho_tich_chung_thuc','SOURCE','UPLOADED')`,
		verID, docID, procID); err != nil {
		t.Fatal(err)
	}
	insertGen := func(jobID, genID uuid.UUID, status, payload string) {
		t.Helper()
		if _, err := pool.Exec(ctx, `
			INSERT INTO document_index_jobs (
				id, document_id, xa_id, request_id, payload_hash, status, procedure_version_id, claim_token, finished_at
			) VALUES ($1,$2,'xa_chu_se',$3,$4,'SUCCEEDED',$5,gen_random_uuid(), now())`,
			jobID, docID, uuid.New(), payload, verID); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `
			INSERT INTO document_index_generations (
				id, xa_id, document_id, procedure_version_id, job_id, status,
				pipeline_version, extraction_version, ocr_version, chunk_config_hash,
				embedding_model_id, embedding_revision, embedding_checksum, vector_dimension,
				source_sha256, content_sha256, manifest_hash, chunk_count, vector_count, published_at
			) VALUES (
				$1,'xa_chu_se',$2,$3,$4,$5,
				'p5','p5','p5','p5','e5-small','rev','',384,
				$6,$6,'',1,1, now()
			)`, genID, docID, verID, jobID, status, sum(genID.String())); err != nil {
			t.Fatal(err)
		}
	}
	insertGen(job1, g1, "READY", "g1")
	if _, err := pool.Exec(ctx, `
		UPDATE procedure_version_documents
		SET index_status='READY', active_generation_id=$3
		WHERE procedure_version_id=$1 AND document_id=$2`, verID, docID, g1); err != nil {
		t.Fatal(err)
	}
	insertChunk := func(gen uuid.UUID, text string, idx int) {
		t.Helper()
		if _, err := pool.Exec(ctx, `
			INSERT INTO knowledge_chunks (
				id, xa_id, procedure_id, procedure_version_id, document_id, chunk_index, content,
				generation_id, page_start, page_end, text_sha256, token_count, extraction_source
			) VALUES ($1,'xa_chu_se',$2,$3,$4,$5,$6,$7,3,3,$8,8,'native')`,
			uuid.New(), procID, verID, docID, idx, text, gen, sum(text)); err != nil {
			t.Fatal(err)
		}
	}
	insertChunk(g1, g1Text, 0)
	insertGen(job2, g2, "STAGING", "g2")
	insertChunk(g2, g2Text, 0)
	return docID, g1, g2
}

func cleanupIndexedDoc(pool *pgxpool.Pool, docID uuid.UUID) {
	ctx := context.Background()
	_, _ = pool.Exec(ctx, `DELETE FROM message_citations WHERE generation_id IN (SELECT id FROM document_index_generations WHERE document_id=$1)`, docID)
	_, _ = pool.Exec(ctx, `DELETE FROM session_source_pins WHERE document_id=$1`, docID)
	_, _ = pool.Exec(ctx, `DELETE FROM knowledge_chunks WHERE document_id=$1`, docID)
	_, _ = pool.Exec(ctx, `UPDATE procedure_version_documents SET active_generation_id=NULL WHERE document_id=$1`, docID)
	_, _ = pool.Exec(ctx, `DELETE FROM document_index_generations WHERE document_id=$1`, docID)
	_, _ = pool.Exec(ctx, `DELETE FROM document_index_jobs WHERE document_id=$1`, docID)
	_, _ = pool.Exec(ctx, `DELETE FROM procedure_version_documents WHERE document_id=$1`, docID)
	_, _ = pool.Exec(ctx, `DELETE FROM documents WHERE id=$1`, docID)
}
