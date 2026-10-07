package storage

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNormalizeEndpointAcceptsHostAndURL(t *testing.T) {
	host, secure, err := normalizeEndpoint("https://s3api-eu-ro-1.runpod.io", false)
	if err != nil || host != "s3api-eu-ro-1.runpod.io:443" || !secure {
		t.Fatalf("runpod url: %s %v %v", host, secure, err)
	}
	host, secure, err = normalizeEndpoint("minio:9000", false)
	if err != nil || host != "minio:9000" || secure {
		t.Fatalf("minio: %s %v %v", host, secure, err)
	}
	host, secure, err = normalizeEndpoint("http://127.0.0.1:9000", true)
	if err != nil || host != "127.0.0.1:9000" || secure {
		t.Fatalf("http url: %s %v %v", host, secure, err)
	}
}

func TestRunPodTLSContract(t *testing.T) {
	host, secure, err := normalizeEndpoint("s3api-eu-ro-1.runpod.io", true)
	if err != nil || host != "s3api-eu-ro-1.runpod.io:443" || !secure {
		t.Fatalf("hostname tls: %s %v %v", host, secure, err)
	}
	_, _, err = normalizeEndpoint("s3api-eu-ro-1.runpod.io", false)
	if err == nil || !strings.Contains(err.Error(), "OBJECT_STORAGE_USE_SSL") {
		t.Fatalf("hostname without tls: %v", err)
	}
	host, secure, err = normalizeEndpoint("https://s3api-eu-ro-1.runpod.io", false)
	if err != nil || host != "s3api-eu-ro-1.runpod.io:443" || !secure {
		t.Fatalf("https url: %s %v %v", host, secure, err)
	}
	_, _, err = normalizeEndpoint("http://s3api-eu-ro-1.runpod.io", true)
	if err == nil || !strings.Contains(err.Error(), "TLS") {
		t.Fatalf("http url: %v", err)
	}
	_, err = NewMinIO(MinIOConfig{
		Endpoint: "s3api-eu-ro-1.runpod.io", AccessKey: "placeholder", SecretKey: "placeholder",
		Bucket: "volume", UseSSL: false, Region: "eu-ro-1", AutoCreateBucket: false,
	})
	if err == nil || !strings.Contains(err.Error(), "OBJECT_STORAGE_USE_SSL") {
		t.Fatalf("client without tls: %v", err)
	}
	_, err = NewMinIO(MinIOConfig{
		Endpoint: "https://s3api-eu-ro-1.runpod.io", AccessKey: "placeholder", SecretKey: "placeholder",
		Bucket: "volume", UseSSL: true, AutoCreateBucket: false,
	})
	if err == nil || !strings.Contains(err.Error(), "OBJECT_STORAGE_REGION") {
		t.Fatalf("missing region: %v", err)
	}
	_, err = NewMinIO(MinIOConfig{
		Endpoint: "https://s3api-eu-ro-1.runpod.io", AccessKey: "placeholder", SecretKey: "placeholder",
		Bucket: "volume", UseSSL: true, Region: "eu-ro-1", AutoCreateBucket: true,
	})
	if err == nil || !strings.Contains(err.Error(), "AUTO_CREATE_BUCKET") {
		t.Fatalf("auto create: %v", err)
	}
	host, secure, err = normalizeEndpoint("minio:9000", false)
	if err != nil || host != "minio:9000" || secure {
		t.Fatalf("local minio: %s %v %v", host, secure, err)
	}
	_, _, err = normalizeEndpoint("https://s3api-eu-ro-1.runpod.io:9000", true)
	if err == nil || !strings.Contains(err.Error(), "443") {
		t.Fatalf("https runpod :9000: %v", err)
	}
	_, _, err = normalizeEndpoint("s3api-eu-ro-1.runpod.io:9000", true)
	if err == nil || !strings.Contains(err.Error(), "443") {
		t.Fatalf("hostname runpod :9000: %v", err)
	}
	host, secure, err = normalizeEndpoint("s3api-eu-ro-1.runpod.io:443", true)
	if err != nil || host != "s3api-eu-ro-1.runpod.io:443" || !secure {
		t.Fatalf("explicit 443: %s %v %v", host, secure, err)
	}
}

func TestRunPodRequiresRegionAndRefusesCreate(t *testing.T) {
	_, err := NewMinIO(MinIOConfig{
		Endpoint:         "https://s3api-eu-ro-1.runpod.io",
		AccessKey:        "placeholder",
		SecretKey:        "placeholder",
		Bucket:           "volume-id",
		UseSSL:           true,
		AutoCreateBucket: false,
	})
	if err == nil || !strings.Contains(err.Error(), "OBJECT_STORAGE_REGION") {
		t.Fatalf("missing region: %v", err)
	}
	_, err = NewMinIO(MinIOConfig{
		Endpoint:         "https://s3api-eu-ro-1.runpod.io",
		AccessKey:        "placeholder",
		SecretKey:        "placeholder",
		Bucket:           "volume-id",
		Region:           "eu-ro-1",
		AutoCreateBucket: true,
	})
	if err == nil || !strings.Contains(err.Error(), "AUTO_CREATE_BUCKET") {
		t.Fatalf("auto create: %v", err)
	}
}

func TestRegionSkipsLocationAndCreate(t *testing.T) {
	var seen []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = append(seen, r.Method+" "+r.URL.RequestURI())
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	endpoint := strings.TrimPrefix(server.URL, "http://")
	client, err := NewMinIO(MinIOConfig{
		Endpoint:         endpoint,
		AccessKey:        "placeholder-access",
		SecretKey:        "placeholder-secret",
		Bucket:           "network-volume",
		Region:           "eu-ro-1",
		AutoCreateBucket: false,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := client.EnsureBucket(context.Background()); err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(seen, "\n")
	if strings.Contains(joined, "location") || strings.Contains(joined, "PUT ") {
		t.Fatalf("unexpected calls: %s", joined)
	}
	if strings.Contains(joined, "placeholder-secret") {
		t.Fatal("secret leaked into the request log")
	}
	_, _ = io.ReadAll(strings.NewReader(joined))
}

func TestLocalAutoCreateIssuesPutWhenMissing(t *testing.T) {
	var seen []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = append(seen, r.Method+" "+r.URL.RequestURI())
		if strings.Contains(r.URL.RawQuery, "location") {
			w.Header().Set("Content-Type", "application/xml")
			_, _ = w.Write([]byte(`<?xml version="1.0" encoding="UTF-8"?><LocationConstraint xmlns="http://s3.amazonaws.com/doc/2006-03-01/"></LocationConstraint>`))
			return
		}
		if r.Method == http.MethodHead {
			http.Error(w, "missing", http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	endpoint := strings.TrimPrefix(server.URL, "http://")
	client, err := NewMinIO(MinIOConfig{
		Endpoint:         endpoint,
		AccessKey:        "placeholder-access",
		SecretKey:        "placeholder-secret",
		Bucket:           "cas-documents",
		AutoCreateBucket: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := client.EnsureBucket(context.Background()); err != nil {
		t.Fatalf("ensure: %v calls=%v", err, seen)
	}
	joined := strings.Join(seen, "\n")
	if !strings.Contains(joined, "PUT ") {
		t.Fatalf("expected bucket create, saw %s", joined)
	}
}
