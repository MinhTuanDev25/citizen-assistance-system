package storage

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/url"
	"strings"
	"time"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

// MinIO talks to a private S3-compatible bucket. Errors returned to callers
// do not include credentials or presigned URLs.
type MinIO struct {
	client *minio.Client
	bucket string
}

type MinIOConfig struct {
	Endpoint  string
	AccessKey string
	SecretKey string
	Bucket    string
	UseSSL    bool
}

func NewMinIO(cfg MinIOConfig) (*MinIO, error) {
	endpoint := strings.TrimSpace(cfg.Endpoint)
	endpoint = strings.TrimPrefix(endpoint, "https://")
	endpoint = strings.TrimPrefix(endpoint, "http://")
	endpoint = strings.TrimRight(endpoint, "/")
	if endpoint == "" || strings.TrimSpace(cfg.Bucket) == "" {
		return nil, fmt.Errorf("object storage endpoint and bucket are required")
	}
	if _, _, err := net.SplitHostPort(endpoint); err != nil {
		return nil, fmt.Errorf("object storage endpoint must be host:port")
	}
	client, err := minio.New(endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(cfg.AccessKey, cfg.SecretKey, ""),
		Secure: cfg.UseSSL,
	})
	if err != nil {
		return nil, fmt.Errorf("object storage client")
	}
	return &MinIO{client: client, bucket: cfg.Bucket}, nil
}

func (m *MinIO) EnsureBucket(ctx context.Context) error {
	exists, err := m.client.BucketExists(ctx, m.bucket)
	if err != nil {
		return fmt.Errorf("object storage bucket check failed")
	}
	if exists {
		return nil
	}
	if err := m.client.MakeBucket(ctx, m.bucket, minio.MakeBucketOptions{}); err != nil {
		return fmt.Errorf("object storage bucket create failed")
	}
	return nil
}

func (m *MinIO) Put(ctx context.Context, key string, r io.Reader, size int64, contentType string) error {
	_, err := m.client.PutObject(ctx, m.bucket, key, r, size, minio.PutObjectOptions{
		ContentType: contentType,
	})
	if err != nil {
		return fmt.Errorf("object storage put failed")
	}
	return nil
}

func (m *MinIO) Get(ctx context.Context, key string) (io.ReadCloser, error) {
	obj, err := m.client.GetObject(ctx, m.bucket, key, minio.GetObjectOptions{})
	if err != nil {
		return nil, fmt.Errorf("object storage get failed")
	}
	if _, err := obj.Stat(); err != nil {
		_ = obj.Close()
		if minio.ToErrorResponse(err).Code == "NoSuchKey" {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("object storage get failed")
	}
	return obj, nil
}

func (m *MinIO) Delete(ctx context.Context, key string) error {
	err := m.client.RemoveObject(ctx, m.bucket, key, minio.RemoveObjectOptions{})
	if err != nil {
		return fmt.Errorf("object storage delete failed")
	}
	return nil
}

func (m *MinIO) PresignGet(ctx context.Context, key string, ttl time.Duration) (string, error) {
	if ttl <= 0 || ttl > 15*time.Minute {
		ttl = 2 * time.Minute
	}
	u, err := m.client.PresignedGetObject(ctx, m.bucket, key, ttl, url.Values{})
	if err != nil {
		return "", fmt.Errorf("object storage presign failed")
	}
	return u.String(), nil
}
