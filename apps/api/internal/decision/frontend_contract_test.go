package decision_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Source-level Phase 1–3 contracts for the web app (no Node required).
func TestFrontendUsesTurnsNotMessageWrite(t *testing.T) {
	root := filepath.Join("..", "..", "..", "web", "src")
	sessionJS, err := os.ReadFile(filepath.Join(root, "api", "session.js"))
	if err != nil {
		t.Fatal(err)
	}
	src := string(sessionJS)
	if !strings.Contains(src, "/turns") {
		t.Fatal("session.js must call /turns")
	}
	if strings.Contains(src, "method: 'POST'") && strings.Contains(src, "/messages") {
		// postUserMessage must not be a live write path
		if !strings.Contains(src, "deprecated") {
			t.Fatal("POST /messages write must be deprecated")
		}
	}
	chatPage, err := os.ReadFile(filepath.Join(root, "pages", "CitizenChatPage.jsx"))
	if err != nil {
		t.Fatal(err)
	}
	page := string(chatPage)
	if !strings.Contains(page, "executeChatTurn") && !strings.Contains(page, "postTurn") {
		t.Fatal("CitizenChatPage must use executeChatTurn/postTurn")
	}
	if strings.Contains(page, "postUserMessage") {
		t.Fatal("CitizenChatPage must not call postUserMessage")
	}
	if strings.Contains(page, "FALLBACK_SUGGESTIONS") || strings.Contains(page, "ho_tich_chung_thuc") {
		t.Fatal("CitizenChatPage must not hardcode domain/fallback procedure names")
	}
}

func TestProductionAppDoesNotStaticallyImportMockStore(t *testing.T) {
	root := filepath.Join("..", "..", "..", "web", "src")
	app, err := os.ReadFile(filepath.Join(root, "App.jsx"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(app), "mockStore") {
		t.Fatal("App.jsx must not reference mockStore")
	}
	if !strings.Contains(string(app), "adminIngestionEnabled") && !strings.Contains(string(app), "DeferredIngestionPage") {
		t.Fatal("App.jsx must gate admin ingestion")
	}
	// Entry pages that are always loaded must not import mockStore
	for _, rel := range []string{
		"pages/CitizenChatPage.jsx",
		"pages/admin/DeferredIngestionPage.jsx",
		"pages/admin/ProceduresPage.jsx",
		"main.jsx",
	} {
		b, err := os.ReadFile(filepath.Join(root, rel))
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(b), "mockStore") {
			t.Fatalf("%s must not import mockStore", rel)
		}
	}
}
