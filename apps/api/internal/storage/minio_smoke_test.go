//go:build miniosmoke

package storage

import (
	"bytes"
	"context"
	"io"
	"os"
	"testing"
	"time"
)

func TestMinIOSmokePutGet(t *testing.T) {
	endpoint := os.Getenv("OBJECT_STORAGE_ENDPOINT")
	if endpoint == "" {
		t.Fatal("OBJECT_STORAGE_ENDPOINT required")
	}
	client, err := NewMinIO(MinIOConfig{
		Endpoint:  endpoint,
		AccessKey: os.Getenv("OBJECT_STORAGE_ACCESS_KEY"),
		SecretKey: os.Getenv("OBJECT_STORAGE_SECRET_KEY"),
		Bucket:    os.Getenv("OBJECT_STORAGE_BUCKET"),
		UseSSL:    false,
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := client.EnsureBucket(ctx); err != nil {
		t.Fatal(err)
	}
	key := ObjectKey("xa_chu_se", "11111111-1111-1111-1111-111111111111", "abc")
	payload := []byte("%PDF-1.4 smoke")
	if err := client.Put(ctx, key, bytes.NewReader(payload), int64(len(payload)), "application/pdf"); err != nil {
		t.Fatal(err)
	}
	rc, err := client.Get(ctx, key)
	if err != nil {
		t.Fatal(err)
	}
	defer rc.Close()
	got, err := io.ReadAll(rc)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("got %q", got)
	}
	if err := client.Delete(ctx, key); err != nil {
		t.Fatal(err)
	}
	if _, err := client.Get(ctx, key); err != ErrNotFound {
		t.Fatalf("missing object err=%v", err)
	}
}
