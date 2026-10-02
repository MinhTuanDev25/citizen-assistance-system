package storage

import (
	"context"
	"errors"
	"io"
	"time"
)

// ErrNotFound is returned when the object key does not exist.
var ErrNotFound = errors.New("storage: object not found")

// ObjectStore is the document blob backend. Handlers never import a vendor SDK.
type ObjectStore interface {
	EnsureBucket(ctx context.Context) error
	Put(ctx context.Context, key string, r io.Reader, size int64, contentType string) error
	Get(ctx context.Context, key string) (io.ReadCloser, error)
	Delete(ctx context.Context, key string) error
	// PresignGet returns a short-lived URL. Callers must not log it.
	PresignGet(ctx context.Context, key string, ttl time.Duration) (string, error)
}

// ObjectKey is the server-chosen blob name. The original filename is never used.
func ObjectKey(xaID, documentID, checksum string) string {
	return xaID + "/documents/" + documentID + "/" + checksum + ".pdf"
}

// InternalURI is the value stored in documents.storage_uri. It is not a public URL.
func InternalURI(bucket, key string) string {
	return "s3://" + bucket + "/" + key
}
