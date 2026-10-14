package activation

import "errors"

var (
	ErrNotFound          = errors.New("not_found")
	ErrValidation        = errors.New("validation")
	ErrConflict          = errors.New("activation_conflict")
	ErrStale             = errors.New("stale_active_version")
	ErrIdempotency       = errors.New("idempotency_conflict")
	ErrForbidden         = errors.New("forbidden")
	ErrInvalidDefinition = errors.New("invalid_definition")
	ErrNotApproved       = errors.New("version_not_approved")
	ErrNotEligible       = errors.New("not_eligible")
	ErrDuplicateVersion  = errors.New("duplicate_version")
)
