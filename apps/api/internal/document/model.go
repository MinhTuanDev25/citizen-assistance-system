package document

import (
	"time"

	"github.com/google/uuid"
)

const (
	StatusUploaded  = "UPLOADED"
	ValidityPending = "PENDING"
	MimePDF         = "application/pdf"
)

// Document is the admin-facing metadata. storage_uri is never serialized.
type Document struct {
	ID               uuid.UUID `json:"id"`
	XAID             string    `json:"xa_id"`
	DomainID         string    `json:"domain_id"`
	Title            string    `json:"title"`
	DocumentNumber   *string   `json:"document_number,omitempty"`
	Issuer           *string   `json:"issuer,omitempty"`
	Filename         string    `json:"filename"`
	Checksum         string    `json:"checksum"`
	MimeType         string    `json:"mime_type"`
	FileSizeBytes    int64     `json:"file_size_bytes"`
	EffectiveDate    *string   `json:"effective_date,omitempty"`
	ExpireDate       *string   `json:"expire_date,omitempty"`
	IssuedDate       *string   `json:"issued_date,omitempty"`
	ProcessingStatus string    `json:"processing_status"`
	ValidityStatus   string    `json:"validity_status"`
	UploadedBy       uuid.UUID `json:"uploaded_by"`
	CreatedAt        time.Time `json:"created_at"`
	UpdatedAt        time.Time `json:"updated_at"`
	StorageURI       string    `json:"-"`
	ObjectKey        string    `json:"-"`
}

type ListFilter struct {
	XAID             string
	DomainID         string
	ProcessingStatus string
	ValidityStatus   string
	Limit            int
	Offset           int
}

type ListResult struct {
	Items    []Document `json:"items"`
	Count    int        `json:"count"`
	Limit    int        `json:"limit"`
	Offset   int        `json:"offset"`
	MaxBytes int64      `json:"max_bytes"`
}

type UploadMeta struct {
	Title          string
	DomainID       string
	DocumentNumber string
	Issuer         string
	EffectiveDate  string
	ExpireDate     string
	IssuedDate     string
	ActorUserID    uuid.UUID
	RequestID      uuid.UUID
}
