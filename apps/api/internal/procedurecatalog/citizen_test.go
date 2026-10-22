package procedurecatalog

import "testing"

func TestHiddenFromCitizen(t *testing.T) {
	hidden := []string{"p5_ask_2557aeb9", "P5_pin_abc", "p5_race_1", " p5_firstpin_x "}
	for _, code := range hidden {
		if !HiddenFromCitizen(code) {
			t.Fatalf("%q should be hidden from citizen catalog", code)
		}
	}
	visible := []string{"dk_khai_sinh", "chung_thuc_ban_sao", "dk_bhyt_ho_gia_dinh", "ht_pin_abc", "ht_ask_1", "ht_race_x", "ht_firstpin_y", ""}
	for _, code := range visible {
		if HiddenFromCitizen(code) {
			t.Fatalf("%q must remain citizen-visible", code)
		}
	}
}
