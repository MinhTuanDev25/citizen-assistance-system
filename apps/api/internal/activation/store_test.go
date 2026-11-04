package activation

import (
	"testing"

	"github.com/google/uuid"
)

func TestHashPayloadStableAndDistinct(t *testing.T) {
	a := hashPayload("p", "v", "e")
	b := hashPayload("p", "v", "e")
	if a != b {
		t.Fatal("hash must be stable")
	}
	if hashPayload("p", "v", "e") == hashPayload("p", "v", "") {
		t.Fatal("different expected version must change hash")
	}
}

func TestNextVersionLabelIncrement(t *testing.T) {
	if got := bumpVersion(nil); got != "1.0.0" {
		t.Fatal(got)
	}
	if got := bumpVersion([]string{"1.0.0", "2.1.9"}); got != "2.1.10" {
		t.Fatal(got)
	}
	// Labels the old draft activation derived from the draft id do not set the sequence.
	if got := bumpVersion([]string{"1.0.0", "8.56948.589", "1.0.1"}); got != "1.0.2" {
		t.Fatal(got)
	}
	if got := bumpVersion([]string{"8.56948.589"}); got != "1.0.0" {
		t.Fatal(got)
	}
}

func TestActivePointerMismatch(t *testing.T) {
	id := uuid.MustParse("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
	other := uuid.MustParse("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
	if !activePointerMismatch(nil, id) {
		t.Fatal("nil pointer vs ACTIVE version is inconsistent")
	}
	if !activePointerMismatch(&other, id) {
		t.Fatal("pointer to another version is inconsistent")
	}
	if activePointerMismatch(&id, id) {
		t.Fatal("matching pointer is the true no-op path")
	}
}
