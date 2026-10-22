package drafting

import (
	"testing"
	"time"
)

func TestDraftTimeoutIsNotExtractCap(t *testing.T) {
	if HardMaxDraftTimeout != 120*time.Second {
		t.Fatalf("draft hard max = %s", HardMaxDraftTimeout)
	}
	c := NewClient("http://ai-service", 45*time.Second)
	if c.Timeout != 45*time.Second {
		t.Fatalf("got %s", c.Timeout)
	}
	clamped := NewClient("http://ai-service", 10*time.Minute)
	if clamped.Timeout != DefaultDraftTimeout {
		t.Fatalf("unbounded timeout was not replaced: %s", clamped.Timeout)
	}
}
