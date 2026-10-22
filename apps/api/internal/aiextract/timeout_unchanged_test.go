package aiextract

import (
	"testing"
	"time"
)

func TestExtractHardMaxTimeoutUnchanged(t *testing.T) {
	if HardMaxTimeout != 8*time.Second {
		t.Fatalf("extract HardMaxTimeout changed: %s", HardMaxTimeout)
	}
	c := NewClient("http://ai-service", 45*time.Second)
	if c.Timeout != HardMaxTimeout {
		t.Fatalf("extract client must still clamp 45s to 8s, got %s", c.Timeout)
	}
}
