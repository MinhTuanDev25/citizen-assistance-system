package speech

import (
	"context"
	"testing"
	"time"
)

func TestValidateSpeechTestDSNRefusesRemote(t *testing.T) {
	t.Setenv("CITIZEN_SPEECH_TEST_ALLOW_REMOTE", "")
	if err := validateSpeechTestDSN("postgres://cas:cas@db.prod.internal:5432/citizen_assistance"); err == nil {
		t.Fatal("expected remote DSN to be refused")
	}
	if err := validateSpeechTestDSN("postgres://cas:cas@127.0.0.1:5432/citizen_assistance?sslmode=disable"); err != nil {
		t.Fatalf("loopback DSN: %v", err)
	}
	if err := validateSpeechTestDSN("postgres://cas:cas@localhost:5432/cas_speech_test"); err != nil {
		t.Fatalf("localhost DSN: %v", err)
	}
}

func TestSpeechTestDSNSkipsImplicitAppDatabase(t *testing.T) {
	t.Setenv("SPEECH_TEST_DATABASE_URL", "")
	t.Setenv("CITIZEN_SPEECH_TEST_ALLOW_DSN", "")
	t.Setenv("DATABASE_URL", "postgres://cas:cas@127.0.0.1:5432/citizen_assistance?sslmode=disable")
	skipped := false
	t.Run("implicit", func(st *testing.T) {
		defer func() { skipped = st.Skipped() }()
		_ = speechTestDSN(st)
		st.Fatal("must not use DATABASE_URL without CITIZEN_SPEECH_TEST_ALLOW_DSN=1")
	})
	if !skipped {
		t.Fatal("expected Skip when SPEECH_TEST_DATABASE_URL is unset")
	}
}

func TestSpeechTestDialFailsWhenDSNSetButUnreachable(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, err := dialSpeechTestPool(ctx, "postgres://cas:cas@127.0.0.1:1/cas_missing?sslmode=disable")
	if err == nil {
		t.Fatal("expected unreachable loopback DSN to return an error, not succeed")
	}
}
