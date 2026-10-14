package activation

import "testing"

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
}
