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
	client     *minio.Client
	bucket     string
	autoCreate bool
}

type MinIOConfig struct {
	Endpoint         string
	AccessKey        string
	SecretKey        string
	Bucket           string
	UseSSL           bool
	Region           string
	AutoCreateBucket bool
}

func NewMinIO(cfg MinIOConfig) (*MinIO, error) {
	endpoint, secure, err := normalizeEndpoint(cfg.Endpoint, cfg.UseSSL)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(cfg.Bucket) == "" {
		return nil, fmt.Errorf("object storage endpoint and bucket are required")
	}
	if isRunPod(endpoint) || isRunPod(cfg.Endpoint) {
		if err := runPodTLSError(cfg.Endpoint, cfg.UseSSL); err != nil {
			return nil, err
		}
		if strings.TrimSpace(cfg.Region) == "" {
			return nil, fmt.Errorf("OBJECT_STORAGE_REGION is required for RunPod object storage")
		}
		if cfg.AutoCreateBucket {
			return nil, fmt.Errorf("OBJECT_STORAGE_AUTO_CREATE_BUCKET must be false for RunPod object storage")
		}
	}
	client, err := minio.New(endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(cfg.AccessKey, cfg.SecretKey, ""),
		Secure: secure,
		Region: strings.TrimSpace(cfg.Region),
	})
	if err != nil {
		return nil, fmt.Errorf("object storage client")
	}
	return &MinIO{client: client, bucket: cfg.Bucket, autoCreate: cfg.AutoCreateBucket}, nil
}

func runPodTLSError(endpoint string, useSSL bool) error {
	if !isRunPod(endpoint) {
		return nil
	}
	lower := strings.ToLower(strings.TrimSpace(endpoint))
	if strings.HasPrefix(lower, "http://") {
		return fmt.Errorf("RunPod object storage requires TLS")
	}
	if strings.HasPrefix(lower, "https://") {
		return nil
	}
	if !useSSL {
		return fmt.Errorf("OBJECT_STORAGE_USE_SSL must be true for RunPod object storage")
	}
	return nil
}

func normalizeEndpoint(raw string, useSSL bool) (string, bool, error) {
	text := strings.TrimSpace(raw)
	if text == "" {
		return "", false, fmt.Errorf("object storage endpoint and bucket are required")
	}
	if err := runPodTLSError(text, useSSL); err != nil {
		return "", false, err
	}
	secure := useSSL
	runpod := isRunPod(text)
	if strings.Contains(text, "://") {
		parsed, err := url.Parse(text)
		if err != nil || parsed.Hostname() == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
			return "", false, fmt.Errorf("object storage endpoint is invalid")
		}
		secure = parsed.Scheme == "https"
		host := parsed.Hostname()
		port := parsed.Port()
		if port == "" {
			if secure || runpod {
				port = "443"
			} else {
				port = "9000"
			}
		}
		if runpod {
			if port != "443" {
				return "", false, fmt.Errorf("RunPod object storage requires port 443")
			}
			secure = true
		}
		return net.JoinHostPort(host, port), secure, nil
	}
	text = strings.TrimRight(text, "/")
	if _, port, err := net.SplitHostPort(text); err != nil {
		added := "9000"
		if secure || runpod {
			added = "443"
		}
		text = net.JoinHostPort(text, added)
	} else if runpod && port != "443" {
		return "", false, fmt.Errorf("RunPod object storage requires port 443")
	}
	if runpod {
		secure = true
	}
	return text, secure, nil
}

func isRunPod(endpoint string) bool {
	return strings.Contains(strings.ToLower(endpoint), "runpod")
}

func (m *MinIO) EnsureBucket(ctx context.Context) error {
	exists, err := m.client.BucketExists(ctx, m.bucket)
	if err != nil {
		return fmt.Errorf("object storage bucket check failed")
	}
	if exists {
		return nil
	}
	if !m.autoCreate {
		return fmt.Errorf("object storage bucket is missing")
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
