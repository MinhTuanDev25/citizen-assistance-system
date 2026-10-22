package procedurecatalog

import "strings"

// HiddenFromCitizen reports leftover p5_* research fixtures that must not
// appear on the citizen catalog or in new intent matching. Existing pinned
// sessions may still resolve those versions through the pin path.
//
// Chat retrieve pin tests must use a non-p5_ procedure_code (for example
// ht_pin_*) or LockTurn will drop the fixture and the unique intent token
// will OUT_OF_SCOPE against the seed catalog only.
func HiddenFromCitizen(code string) bool {
	c := strings.ToLower(strings.TrimSpace(code))
	return strings.HasPrefix(c, "p5_")
}
