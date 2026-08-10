package config

import (
	"path/filepath"
	"testing"
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
}
