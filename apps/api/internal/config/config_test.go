package config

import (
	"path/filepath"
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
