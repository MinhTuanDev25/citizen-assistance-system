package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Config holds runtime settings (YAML per env + env overrides).
// Bake YAML into image; inject secrets via env at deploy (Compose/VPS).
type Config struct {
	Env             string
	APIAddr         string
	DatabaseURL     string
	XAID            string
	LogLevel        string
	DBMaxConns      int32
	DBMinConns      int32
	DBConnTimeout   time.Duration
	ShutdownTimeout time.Duration
	ConfigFile      string
}

type fileConfig struct {
	Server struct {
		Addr               string `yaml:"addr"`
		ShutdownTimeoutSec int    `yaml:"shutdown_timeout_sec"`
	} `yaml:"server"`
	Database struct {
		URL            string `yaml:"url"`
		MaxConns       int    `yaml:"max_conns"`
		MinConns       int    `yaml:"min_conns"`
		ConnTimeoutSec int    `yaml:"conn_timeout_sec"`
	} `yaml:"database"`
	App struct {
		XAID     string `yaml:"xa_id"`
		LogLevel string `yaml:"log_level"`
	} `yaml:"app"`
}

// Load reads configs/{APP_ENV}/config.yaml then applies environment overrides.
// Precedence: env vars > YAML > built-in defaults.
func Load() (Config, error) {
	envName := strings.ToLower(strings.TrimSpace(os.Getenv("APP_ENV")))
	if envName == "" {
		envName = "local"
	}
	if envName != "local" && envName != "prod" {
		return Config{}, fmt.Errorf("APP_ENV must be local or prod, got %q", envName)
	}

	path, err := resolveConfigPath(envName)
	if err != nil {
		return Config{}, err
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("read config %s: %w", path, err)
	}

	var fc fileConfig
	if err := yaml.Unmarshal(raw, &fc); err != nil {
		return Config{}, fmt.Errorf("parse config %s: %w", path, err)
	}

	cfg := Config{
		Env:             envName,
		ConfigFile:      path,
		APIAddr:         firstNonEmpty(fc.Server.Addr, ":8080"),
		DatabaseURL:     strings.TrimSpace(fc.Database.URL),
		XAID:            firstNonEmpty(fc.App.XAID, "xa_chu_se"),
		LogLevel:        firstNonEmpty(fc.App.LogLevel, "info"),
		DBMaxConns:      int32(defaultInt(fc.Database.MaxConns, 10)),
		DBMinConns:      int32(defaultInt(fc.Database.MinConns, 1)),
		DBConnTimeout:   time.Duration(defaultInt(fc.Database.ConnTimeoutSec, 5)) * time.Second,
		ShutdownTimeout: time.Duration(defaultInt(fc.Server.ShutdownTimeoutSec, 10)) * time.Second,
	}

	applyEnvOverrides(&cfg)

	if cfg.DatabaseURL == "" {
		return Config{}, fmt.Errorf("DATABASE_URL missing for APP_ENV=%s (set env or database.url in YAML)", envName)
	}
	if cfg.DBMaxConns < 1 {
		return Config{}, fmt.Errorf("database.max_conns must be >= 1")
	}
	if cfg.DBMinConns < 0 || cfg.DBMinConns > cfg.DBMaxConns {
		return Config{}, fmt.Errorf("database.min_conns must be between 0 and max_conns")
	}
	return cfg, nil
}

func applyEnvOverrides(cfg *Config) {
	if v := os.Getenv("API_ADDR"); v != "" {
		cfg.APIAddr = v
	}
	if v := os.Getenv("DATABASE_URL"); v != "" {
		cfg.DatabaseURL = v
	}
	if v := os.Getenv("XA_ID"); v != "" {
		cfg.XAID = v
	}
	if v := os.Getenv("LOG_LEVEL"); v != "" {
		cfg.LogLevel = v
	}
	if v := os.Getenv("DB_MAX_CONNS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.DBMaxConns = int32(n)
		}
	}
	if v := os.Getenv("DB_MIN_CONNS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.DBMinConns = int32(n)
		}
	}
	if v := os.Getenv("DB_CONN_TIMEOUT_SEC"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.DBConnTimeout = time.Duration(n) * time.Second
		}
	}
	if v := os.Getenv("SHUTDOWN_TIMEOUT_SEC"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.ShutdownTimeout = time.Duration(n) * time.Second
		}
	}
}

func resolveConfigPath(envName string) (string, error) {
	if p := os.Getenv("CONFIG_FILE"); p != "" {
		return p, nil
	}

	rel := filepath.Join("configs", envName, "config.yaml")
	candidates := []string{
		rel,
		filepath.Join("apps", "api", rel),
	}

	for _, p := range candidates {
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			abs, err := filepath.Abs(p)
			if err != nil {
				return p, nil
			}
			return abs, nil
		}
	}
	return "", fmt.Errorf(
		"config file configs/%s/config.yaml not found (run from apps/api with APP_ENV=%s, or set CONFIG_FILE)",
		envName, envName,
	)
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func defaultInt(v, fallback int) int {
	if v == 0 {
		return fallback
	}
	return v
}
