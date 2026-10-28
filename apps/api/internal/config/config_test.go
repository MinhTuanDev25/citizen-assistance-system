package config

import (
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestLoadLocal(t *testing.T) {
	t.Setenv("APP_ENV", "local")
	t.Setenv("CONFIG_FILE", filepath.Join("..", "..", "configs", "local", "config.yaml"))
	t.Setenv("DATABASE_URL", "")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Env != "local" {
		t.Fatalf("env=%s", cfg.Env)
	}
	if cfg.DatabaseURL == "" {
		t.Fatal("expected database url from local yaml")
	}
}

func TestLoadProdRequiresSecret(t *testing.T) {
	t.Setenv("APP_ENV", "prod")
	t.Setenv("CONFIG_FILE", filepath.Join("..", "..", "configs", "prod", "config.yaml"))
	t.Setenv("DATABASE_URL", "")
	_, err := Load()
	if err == nil {
		t.Fatal("expected error when prod DATABASE_URL missing")
	}
}

func TestLoadProdWithURL(t *testing.T) {
	t.Setenv("APP_ENV", "prod")
	t.Setenv("CONFIG_FILE", filepath.Join("..", "..", "configs", "prod", "config.yaml"))
	t.Setenv("DATABASE_URL", "postgres://u:p@h:5432/db?sslmode=require")
	t.Setenv("JWT_SECRET", "prod-jwt-secret-16chars")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.DatabaseURL != "postgres://u:p@h:5432/db?sslmode=require" {
		t.Fatalf("url=%s", cfg.DatabaseURL)
	}
	if cfg.JWTSecret != "prod-jwt-secret-16chars" {
		t.Fatalf("jwt=%s", cfg.JWTSecret)
	}
	if cfg.AIExtractEnabled {
		t.Fatal("AI extract must stay disabled by default")
	}
}

func TestLoadProdRejectsLocalDatabaseAndIndexing(t *testing.T) {
	t.Setenv("APP_ENV", "prod")
	t.Setenv("CONFIG_FILE", filepath.Join("..", "..", "configs", "prod", "config.yaml"))
	t.Setenv("JWT_SECRET", "prod-jwt-secret-16chars")
	t.Setenv("DATABASE_URL", "postgres://u:p@127.0.0.1:5432/citizen_assistance?sslmode=disable")
	_, err := Load()
	if err == nil || !strings.Contains(err.Error(), "sslmode") {
		t.Fatalf("sslmode disable accepted: %v", err)
	}
	t.Setenv("DATABASE_URL", "postgres://u:p@db.internal:5432/citizen_assistance?sslmode=require")
	t.Setenv("JWT_SECRET", "local-dev-jwt-secret-change-me")
	_, err = Load()
	if err == nil || !strings.Contains(err.Error(), "JWT_SECRET") {
		t.Fatalf("dev jwt accepted: %v", err)
	}
	t.Setenv("JWT_SECRET", "prod-jwt-secret-16chars")
	t.Setenv("ADMIN_INGESTION_ENABLED", "true")
	t.Setenv("ADMIN_INDEXING_ENABLED", "true")
	t.Setenv("OBJECT_STORAGE_ENDPOINT", "storage.internal:443")
	t.Setenv("OBJECT_STORAGE_ACCESS_KEY", "prod-access")
	t.Setenv("OBJECT_STORAGE_SECRET_KEY", "prod-secret-value")
	t.Setenv("OBJECT_STORAGE_BUCKET", "cas-documents")
	t.Setenv("OBJECT_STORAGE_USE_SSL", "true")
	t.Setenv("OBJECT_STORAGE_AUTO_CREATE_BUCKET", "false")
	t.Setenv("DOCUMENT_MAX_BYTES", "20971520")
	t.Setenv("AI_SERVICE_URL", "http://ai-service:8001")
	t.Setenv("AI_SERVICE_TOKEN", "prod-ai-service-token")
	t.Setenv("INDEX_MODE", "mock")
	_, err = Load()
	if err == nil || !strings.Contains(err.Error(), "INDEX_MODE") {
		t.Fatalf("mock indexing accepted: %v", err)
	}
}

func TestLoadRejectsBadAIConfig(t *testing.T) {
	t.Setenv("APP_ENV", "local")
	t.Setenv("CONFIG_FILE", filepath.Join("..", "..", "configs", "local", "config.yaml"))
	cases := []struct {
		key, val string
	}{
		{"AI_EXTRACT_TIMEOUT_MS", "0"},
		{"AI_EXTRACT_TIMEOUT_MS", "9000"},
		{"AI_EXTRACT_TIMEOUT_MS", "abc"},
		{"AI_INTENT_SELECT_MIN", "-1"},
		{"AI_INTENT_CONFIRM_MIN", "0.9"},
		{"AI_SLOT_CONFIDENCE_MIN", "2"},
		{"AI_SLOT_CONFIDENCE_MIN", "NaN"},
		{"AI_INTENT_SELECT_MIN", "+Inf"},
		{"AI_INTENT_CONFIRM_MIN", "-Inf"},
		{"AI_EXTRACT_ENABLED", "maybe"},
		{"AI_RETRIEVE_ENABLED", "maybe"},
		{"AI_RETRIEVE_TIMEOUT_MS", "0"},
		{"AI_RETRIEVE_TIMEOUT_MS", "11000"},
		{"AI_DRAFT_TIMEOUT_MS", "0"},
		{"AI_DRAFT_TIMEOUT_MS", "310001"},
		{"AI_DRAFT_TIMEOUT_MS", "abc"},
	}
	for _, tc := range cases {
		t.Run(tc.key+"="+tc.val, func(t *testing.T) {
			t.Setenv(tc.key, tc.val)
			if tc.key == "AI_INTENT_CONFIRM_MIN" {
				t.Setenv("AI_INTENT_SELECT_MIN", "0.5")
			}
			_, err := Load()
			if err == nil {
				t.Fatal("expected config error")
			}
			if len(tc.val) > 2 && strings.Contains(err.Error(), tc.val) {
				t.Fatalf("error echoed config value %q: %v", tc.val, err)
			}
		})
	}
}

func TestLoadAcceptsRetrieveTimeoutTenSeconds(t *testing.T) {
	t.Setenv("APP_ENV", "local")
	t.Setenv("CONFIG_FILE", filepath.Join("..", "..", "configs", "local", "config.yaml"))
	t.Setenv("AI_RETRIEVE_TIMEOUT_MS", "10000")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.AIRetrieveTimeoutMS != 10000 {
		t.Fatalf("AIRetrieveTimeoutMS=%d", cfg.AIRetrieveTimeoutMS)
	}
}

func TestLoadRequiresTokenWhenRetrieveEnabled(t *testing.T) {
	t.Setenv("APP_ENV", "local")
	t.Setenv("CONFIG_FILE", filepath.Join("..", "..", "configs", "local", "config.yaml"))
	t.Setenv("AI_RETRIEVE_ENABLED", "true")
	t.Setenv("AI_SERVICE_URL", "http://ai-service:8001")
	t.Setenv("AI_SERVICE_TOKEN", "")
	if _, err := Load(); err == nil {
		t.Fatal("expected missing token error")
	}
}

func TestLoadRequiresTokenWhenAIEnabled(t *testing.T) {
	t.Setenv("APP_ENV", "local")
	t.Setenv("CONFIG_FILE", filepath.Join("..", "..", "configs", "local", "config.yaml"))
	t.Setenv("AI_EXTRACT_ENABLED", "true")
	t.Setenv("AI_SERVICE_URL", "http://ai-service:8001")
	t.Setenv("AI_SERVICE_TOKEN", "")
	if _, err := Load(); err == nil {
		t.Fatal("expected missing token error")
	}
}

func TestLoadRejectsShortServiceTokenWithoutEcho(t *testing.T) {
	t.Setenv("APP_ENV", "local")
	t.Setenv("CONFIG_FILE", filepath.Join("..", "..", "configs", "local", "config.yaml"))
	t.Setenv("AI_EXTRACT_ENABLED", "false")
	secret := "too-short-value"
	t.Setenv("AI_SERVICE_TOKEN", secret)
	_, err := Load()
	if err == nil {
		t.Fatal("expected short token error")
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("error echoed token: %v", err)
	}
}

func TestLoadIngestionDisabledIgnoresStorage(t *testing.T) {
	t.Setenv("APP_ENV", "local")
	t.Setenv("CONFIG_FILE", filepath.Join("..", "..", "configs", "local", "config.yaml"))
	t.Setenv("ADMIN_INGESTION_ENABLED", "false")
	t.Setenv("OBJECT_STORAGE_ENDPOINT", "")
	t.Setenv("OBJECT_STORAGE_ACCESS_KEY", "")
	t.Setenv("OBJECT_STORAGE_SECRET_KEY", "")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.AdminIngestionEnabled {
		t.Fatal("ingestion should stay off")
	}
}

func TestLoadIngestionRequiresStorageWithoutEcho(t *testing.T) {
	t.Setenv("APP_ENV", "local")
	t.Setenv("CONFIG_FILE", filepath.Join("..", "..", "configs", "local", "config.yaml"))
	t.Setenv("ADMIN_INGESTION_ENABLED", "true")
	secret := "object-secret-value"
	t.Setenv("OBJECT_STORAGE_SECRET_KEY", secret)
	t.Setenv("OBJECT_STORAGE_ENDPOINT", "")
	t.Setenv("OBJECT_STORAGE_ACCESS_KEY", "")
	t.Setenv("OBJECT_STORAGE_BUCKET", "")
	_, err := Load()
	if err == nil {
		t.Fatal("expected missing storage config")
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("error echoed secret: %v", err)
	}
}

func TestSpeechDisabledByDefault(t *testing.T) {
	t.Setenv("APP_ENV", "local")
	t.Setenv("CONFIG_FILE", filepath.Join("..", "..", "configs", "local", "config.yaml"))
	t.Setenv("DATABASE_URL", "")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.SpeechEnabled || cfg.SpeechServiceURL != "" || cfg.SpeechServiceToken != "" {
		t.Fatalf("speech default %+v", cfg)
	}
	if cfg.SpeechTimeoutMS != 130000 || cfg.SpeechMaxUploadBytes != SpeechUploadHardMaxBytes || cfg.SpeechMaxInflight != 1 {
		t.Fatalf("speech bounds %+v", cfg)
	}
}

func TestSpeechMaxUploadBytesBounds(t *testing.T) {
	base := Config{
		SpeechTimeoutMS: 130000, SpeechMaxInflight: 1, SpeechEnabled: false,
	}
	for _, n := range []int64{0, 1, 65535, 65536, SpeechUploadHardMaxBytes + 1} {
		cfg := base
		cfg.SpeechMaxUploadBytes = n
		if err := validateSpeechConfig(cfg); err == nil {
			t.Fatalf("%d accepted", n)
		}
	}
	for _, n := range []int64{65537, SpeechUploadMinBytes, SpeechUploadHardMaxBytes} {
		cfg := base
		cfg.SpeechMaxUploadBytes = n
		if err := validateSpeechConfig(cfg); err != nil {
			t.Fatalf("%d rejected: %v", n, err)
		}
	}

	t.Setenv("APP_ENV", "local")
	t.Setenv("CONFIG_FILE", filepath.Join("..", "..", "configs", "local", "config.yaml"))
	t.Setenv("DATABASE_URL", "")
	t.Setenv("SPEECH_MAX_UPLOAD_BYTES", "65536")
	if _, err := Load(); err == nil || !strings.Contains(err.Error(), "SPEECH_MAX_UPLOAD_BYTES") {
		t.Fatalf("env 65536: %v", err)
	}
	t.Setenv("SPEECH_MAX_UPLOAD_BYTES", "65537")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.SpeechMaxUploadBytes != 65537 {
		t.Fatalf("got %d", cfg.SpeechMaxUploadBytes)
	}
	t.Setenv("SPEECH_MAX_UPLOAD_BYTES", strconv.FormatInt(SpeechUploadHardMaxBytes, 10))
	cfg, err = Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.SpeechMaxUploadBytes != SpeechUploadHardMaxBytes {
		t.Fatalf("default hard max via env: %d", cfg.SpeechMaxUploadBytes)
	}
}

func TestSpeechEnabledFailClosed(t *testing.T) {
	t.Setenv("APP_ENV", "local")
	t.Setenv("CONFIG_FILE", filepath.Join("..", "..", "configs", "local", "config.yaml"))
	t.Setenv("SPEECH_ENABLED", "true")
	_, err := Load()
	if err == nil || !strings.Contains(err.Error(), "SPEECH_SERVICE_URL") {
		t.Fatalf("empty url: %v", err)
	}
	t.Setenv("SPEECH_SERVICE_URL", "http://127.0.0.1:8090")
	_, err = Load()
	if err == nil || !strings.Contains(err.Error(), "SPEECH_SERVICE_TOKEN") {
		t.Fatalf("empty token: %v", err)
	}
	t.Setenv("SPEECH_SERVICE_TOKEN", "speech-service-token-1")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.SpeechEnabled || cfg.SpeechServiceURL != "http://127.0.0.1:8090" {
		t.Fatalf("%+v", cfg)
	}
	for _, raw := range []string{
		"http://127.0.0.1:18090",
		"http://127.0.0.1:8090",
		"http://localhost:8090",
		"http://[::1]:8090",
	} {
		t.Setenv("SPEECH_SERVICE_URL", raw)
		cfg, err := Load()
		if err != nil {
			t.Fatalf("%s: %v", raw, err)
		}
		if cfg.SpeechServiceURL != raw {
			t.Fatalf("%s stored %s", raw, cfg.SpeechServiceURL)
		}
	}

	rejects := map[string]string{
		"file:///tmp/x":                   "scheme",
		"https://speech.example.com":      "scheme",
		"http://user:pass@127.0.0.1:8090": "credentials",
		"http://example.com:8090":         "loopback",
		"http://8.8.8.8:8090":             "loopback",
		"http://10.0.0.5:8090":            "loopback",
		"http://speech:8090":              "loopback",
		"http://127.0.0.1:8090/v1":        "path",
		"http://127.0.0.1:8090?x=1":       "query",
		"http://127.0.0.1:8090#frag":      "fragment",
	}
	for raw, want := range rejects {
		t.Setenv("SPEECH_SERVICE_URL", raw)
		_, err := Load()
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("%s: want %q got %v", raw, want, err)
		}
	}

	secret := "speech-token-secret-xx"
	t.Setenv("SPEECH_SERVICE_URL", "http://127.0.0.1:8090")
	t.Setenv("SPEECH_SERVICE_TOKEN", secret)
	t.Setenv("SPEECH_TIMEOUT_MS", "0")
	_, err = Load()
	if err == nil || strings.Contains(err.Error(), secret) {
		t.Fatalf("timeout leaked: %v", err)
	}
}

func TestSpeechProdRejectsPlaceholderToken(t *testing.T) {
	cfg := Config{
		Env: "prod", SpeechEnabled: true,
		DatabaseURL:        "postgres://u:p@h:5432/db?sslmode=require",
		SpeechServiceURL:   "http://127.0.0.1:8090",
		SpeechServiceToken: "local-dev-speech-token",
		SpeechTimeoutMS:    130000, SpeechMaxUploadBytes: SpeechUploadHardMaxBytes, SpeechMaxInflight: 1,
	}
	if err := validateProduction(cfg); err == nil || !strings.Contains(err.Error(), "SPEECH_SERVICE_TOKEN") {
		t.Fatalf("%v", err)
	}
}

func TestIndexingFlagFailClosed(t *testing.T) {
	secret := "index-token-secret-value"
	err := validateIndexingConfig(Config{AdminIndexingEnabled: true, AIServiceToken: secret})
	if err == nil || strings.Contains(err.Error(), secret) {
		t.Fatal(err)
	}
	if err := validateIndexingConfig(Config{}); err != nil {
		t.Fatal(err)
	}
}

func TestIndexLeaseMustExceedTimeout(t *testing.T) {
	base := Config{IndexClaimTTL: 3 * time.Second, IndexTimeout: 3 * time.Second}
	if err := validateIndexLease(base); err == nil {
		t.Fatal("ttl equal to timeout was accepted")
	}
	base.IndexClaimTTL = 4 * time.Second
	if err := validateIndexLease(base); err == nil {
		t.Fatal("ttl equal to timeout plus margin was accepted")
	}
	base.IndexClaimTTL = 5 * time.Second
	if err := validateIndexLease(base); err != nil {
		t.Fatal(err)
	}
}

func TestAsyncIndexLeaseFollowsHeartbeatNotJobLength(t *testing.T) {
	cfg := Config{
		IndexMode: "pipeline", IndexAsync: true,
		IndexPipelineTimeout: time.Hour, IndexHeartbeat: 15 * time.Second,
		IndexClaimTTL: 2 * time.Minute, IndexMaxConcurrent: 1, IndexPublishTimeout: 10 * time.Second,
	}
	if err := validateIndexLease(cfg); err != nil {
		t.Fatalf("a one-hour job with a short renewed lease must be valid: %v", err)
	}
	sync := cfg
	sync.IndexAsync = false
	if err := validateIndexLease(sync); err == nil {
		t.Fatal("without the heartbeat the lease must still outlive the pipeline timeout")
	}
	for name, mutate := range map[string]func(*Config){
		"ttl below 3 beats":   func(c *Config) { c.IndexClaimTTL = 44 * time.Second },
		"timeout above 2h":    func(c *Config) { c.IndexPipelineTimeout = 2*time.Hour + time.Second },
		"heartbeat too short": func(c *Config) { c.IndexHeartbeat = 0 },
		"too many workers":    func(c *Config) { c.IndexMaxConcurrent = 5 },
	} {
		bad := cfg
		mutate(&bad)
		if err := validateIndexLease(bad); err == nil {
			t.Fatalf("%s was accepted", name)
		}
	}
}

func TestIndexAsyncEnvParsing(t *testing.T) {
	t.Setenv("INDEX_ASYNC", "true")
	t.Setenv("INDEX_HEARTBEAT_SECONDS", "20")
	t.Setenv("INDEX_MAX_CONCURRENT", "2")
	cfg := Config{}
	if err := applyEnvOverrides(&cfg); err != nil {
		t.Fatal(err)
	}
	if !cfg.IndexAsync || cfg.IndexHeartbeat != 20*time.Second || cfg.IndexMaxConcurrent != 2 {
		t.Fatalf("parsed %+v", cfg)
	}
	t.Setenv("INDEX_ASYNC", "maybe")
	if err := applyEnvOverrides(&Config{}); err == nil {
		t.Fatal("INDEX_ASYNC=maybe was accepted")
	}
}

func TestRunPodRejectsNonTLSPort(t *testing.T) {
	if err := runPodPortError("https://s3api-eu-ro-1.runpod.io:9000"); err == nil || !strings.Contains(err.Error(), "443") {
		t.Fatalf("https port: %v", err)
	}
	if err := runPodPortError("s3api-eu-ro-1.runpod.io:9000"); err == nil || !strings.Contains(err.Error(), "443") {
		t.Fatalf("hostname port: %v", err)
	}
	if err := runPodPortError("https://s3api-eu-ro-1.runpod.io"); err != nil {
		t.Fatal(err)
	}
	if err := runPodPortError("s3api-eu-ro-1.runpod.io:443"); err != nil {
		t.Fatal(err)
	}
	secret := "super-secret-value"
	err := validateIngestionConfig(Config{
		AdminIngestionEnabled:      true,
		ObjectStorageEndpoint:      "https://s3api-eu-ro-1.runpod.io:9000",
		ObjectStorageAccessKey:     secret,
		ObjectStorageSecretKey:     secret,
		ObjectStorageBucket:        "volume",
		ObjectStorageSSLConfigured: true,
		ObjectStorageUseSSL:        true,
		ObjectStorageRegion:        "eu-ro-1",
	})
	if err == nil || !strings.Contains(err.Error(), "443") || strings.Contains(err.Error(), secret) {
		t.Fatalf("ingestion port: %v", err)
	}
}
