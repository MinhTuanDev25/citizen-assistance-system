package decision

import (
	"testing"

	"golang.org/x/text/unicode/norm"
)

func TestFoldNFCEquivalence(t *testing.T) {
	precomposed := "Đăng ký khai sinh"
	decomposed := norm.NFD.String(precomposed)
	if precomposed == decomposed {
		t.Fatal("expected NFD to differ from NFC source for Vietnamese text")
	}
	if Fold(precomposed) != Fold(decomposed) {
		t.Fatalf("Fold(NFC)=%q Fold(NFD)=%q", Fold(precomposed), Fold(decomposed))
	}
	if Fold(precomposed) != Fold(norm.NFC.String(decomposed)) {
		t.Fatal("NFC round-trip fold mismatch")
	}
}

func TestFoldCaseAndDiacritics(t *testing.T) {
	a := Fold("KHAI SINH")
	b := Fold("khai sinh")
	c := Fold("khải sinh")
	if a != b {
		t.Fatalf("%q != %q", a, b)
	}
	if a != c {
		t.Fatalf("diacritic fold: %q != %q", a, c)
	}
}
