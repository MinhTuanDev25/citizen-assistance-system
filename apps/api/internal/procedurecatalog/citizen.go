package procedurecatalog

import "strings"

// HiddenFromCitizen reports research/integration fixture codes that must not
// appear on the citizen catalog or in new intent matching. Existing pinned
// sessions may still resolve those versions through the pin path.
func HiddenFromCitizen(code string) bool {
	c := strings.ToLower(strings.TrimSpace(code))
	return strings.HasPrefix(c, "p5_")
}
