package document

import "errors"

var (
	ErrTooLarge      = errors.New("document: too large")
	ErrBadPDF        = errors.New("document: invalid pdf")
	ErrValidation    = errors.New("document: validation")
	ErrDuplicate     = errors.New("document: duplicate")
	ErrDomain        = errors.New("document: domain")
	ErrNotFound      = errors.New("document: not found")
	ErrObjectMissing = errors.New("document: object missing")
	ErrStorage       = errors.New("document: storage")
)

type validationError struct{ msg string }

func errValidation(msg string) error { return &validationError{msg: msg} }

func (e *validationError) Error() string { return e.msg }

func (e *validationError) Unwrap() error { return ErrValidation }

// SafeMessage is a client-facing validation string that never includes paths or file bytes.
func SafeMessage(err error) string {
	var v *validationError
	if errors.As(err, &v) && v.msg != "" {
		return v.msg
	}
	return "invalid document"
}
