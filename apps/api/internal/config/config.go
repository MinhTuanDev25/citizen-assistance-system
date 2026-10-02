package config

import (
	"fmt"
	"math"
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
	Env              string
	APIAddr          string
	DatabaseURL      string
	XAID             string
	LogLevel         string
	CitizenDomainIDs []string
	DBMaxConns       int32
	DBMinConns       int32
	DBConnTimeout    time.Duration
	ShutdownTimeout  time.Duration
	ConfigFile       string
	JWTSecret        string
	JWTExpireHours   int

	// AI extraction (P2 — LLM Extract). Disabled by default: the zero value
	// (AIExtractEnabled=false, AIServiceURL="") fully preserves pre-P2
	// keyword-only behavior. See apps/api/README.md and .env.example.
	AIExtractEnabled    bool
	AIServiceURL        string
	AIServiceToken      string
	AIExtractTimeoutMS  int
	AIIntentSelectMin   float64
	AIIntentConfirmMin  float64
	AISlotConfidenceMin float64

	// Admin PDF intake (P3.1). Off by default: the API does not open object storage.
	AdminIngestionEnabled      bool
	AdminIndexingEnabled       bool
	IndexMode                  string
	IndexClaimTTL              time.Duration
	IndexTimeout               time.Duration
	IndexPipelineTimeout       time.Duration
	IndexPublishTimeout        time.Duration
	ObjectStorageEndpoint      string
	ObjectStorageAccessKey     string
	ObjectStorageSecretKey     string
	ObjectStorageBucket        string
	ObjectStorageUseSSL        bool
	ObjectStorageSSLConfigured bool
	DocumentMaxBytes           int64
}

// DocumentMaxHardBytes matches middleware.DocumentUploadHardMax.
const DocumentMaxHardBytes int64 = 64 << 20

// AIExtractTimeoutHardMaxMS matches aiextract.HardMaxTimeout. A configured
// timeout above this is a startup error, not a silent clamp.
const AIExtractTimeoutHardMaxMS = 8000

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
		XAID             string   `yaml:"xa_id"`
		LogLevel         string   `yaml:"log_level"`
		CitizenDomainIDs []string `yaml:"citizen_domain_ids"`
	} `yaml:"app"`
	Auth struct {
		JWTSecret      string `yaml:"jwt_secret"`
		JWTExpireHours int    `yaml:"jwt_expire_hours"`
	} `yaml:"auth"`
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
		Env:              envName,
		ConfigFile:       path,
		APIAddr:          firstNonEmpty(fc.Server.Addr, ":8080"),
		DatabaseURL:      strings.TrimSpace(fc.Database.URL),
		XAID:             firstNonEmpty(fc.App.XAID, "xa_chu_se"),
		LogLevel:         firstNonEmpty(fc.App.LogLevel, "info"),
		CitizenDomainIDs: append([]string{}, fc.App.CitizenDomainIDs...),
		DBMaxConns:       int32(defaultInt(fc.Database.MaxConns, 10)),
		DBMinConns:       int32(defaultInt(fc.Database.MinConns, 1)),
		DBConnTimeout:    time.Duration(defaultInt(fc.Database.ConnTimeoutSec, 5)) * time.Second,
		ShutdownTimeout:  time.Duration(defaultInt(fc.Server.ShutdownTimeoutSec, 10)) * time.Second,
		JWTSecret:        strings.TrimSpace(fc.Auth.JWTSecret),
		JWTExpireHours:   defaultInt(fc.Auth.JWTExpireHours, 24),

		AIExtractEnabled:     false,
		AIServiceURL:         "",
		AIExtractTimeoutMS:   3000,
		AIIntentSelectMin:    0.82,
		AIIntentConfirmMin:   0.55,
		AISlotConfidenceMin:  0.6,
		IndexMode:            "mock",
		IndexClaimTTL:        30 * time.Second,
		IndexTimeout:         3 * time.Second,
		IndexPipelineTimeout: 2 * time.Minute,
		IndexPublishTimeout:  10 * time.Second,
	}

	if err := applyEnvOverrides(&cfg); err != nil {
		return Config{}, err
	}
	if len(cfg.CitizenDomainIDs) == 0 {
		cfg.CitizenDomainIDs = []string{"ho_tich_chung_thuc"}
	}

	if cfg.DatabaseURL == "" {
		return Config{}, fmt.Errorf("DATABASE_URL missing for APP_ENV=%s (set env or database.url in YAML)", envName)
	}
	if cfg.JWTSecret == "" {
		return Config{}, fmt.Errorf("JWT_SECRET missing for APP_ENV=%s (set env or auth.jwt_secret in YAML)", envName)
	}
	if len(cfg.JWTSecret) < 16 {
		return Config{}, fmt.Errorf("JWT_SECRET must be at least 16 characters")
	}
	if cfg.JWTExpireHours < 1 {
		return Config{}, fmt.Errorf("auth.jwt_expire_hours must be >= 1")
	}
	if cfg.DBMaxConns < 1 {
		return Config{}, fmt.Errorf("database.max_conns must be >= 1")
	}
	if cfg.DBMinConns < 0 || cfg.DBMinConns > cfg.DBMaxConns {
		return Config{}, fmt.Errorf("database.min_conns must be between 0 and max_conns")
	}
	if err := validateAIConfig(cfg); err != nil {
		return Config{}, err
	}
	if err := validateIngestionConfig(cfg); err != nil {
		return Config{}, err
	}
	if err := validateIndexingConfig(cfg); err != nil {
		return Config{}, err
	}
	if err := validateIndexLease(cfg); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

// IndexLeaseSafetyMargin keeps a crashed claim recoverable after the worker timeout.
const IndexLeaseSafetyMargin = time.Second

func validateIndexLease(cfg Config) error {
	if cfg.IndexMode == "pipeline" {
		if cfg.IndexPipelineTimeout <= 0 || cfg.IndexPipelineTimeout > 15*time.Minute {
			return fmt.Errorf("INDEX_PIPELINE_TIMEOUT_SECONDS must be in (0, 900]")
		}
		if cfg.IndexClaimTTL <= cfg.IndexPipelineTimeout+IndexLeaseSafetyMargin {
			return fmt.Errorf("INDEX_CLAIM_TTL_SECONDS must be greater than INDEX_PIPELINE_TIMEOUT_SECONDS plus 1s")
		}
		if cfg.IndexClaimTTL > 16*time.Minute {
			return fmt.Errorf("INDEX_CLAIM_TTL_SECONDS must be in (0, 960] when INDEX_MODE=pipeline")
		}
		if cfg.IndexPublishTimeout <= 0 || cfg.IndexPublishTimeout > time.Minute {
			return fmt.Errorf("INDEX_PUBLISH_TIMEOUT_MS must be in (0, 60000]")
		}
		return nil
	}
	if cfg.IndexClaimTTL <= 0 || cfg.IndexClaimTTL > time.Minute {
		return fmt.Errorf("INDEX_CLAIM_TTL_SECONDS must be in (0, 60]")
	}
	if cfg.IndexTimeout <= 0 || cfg.IndexTimeout > 10*time.Second {
		return fmt.Errorf("INDEX_TIMEOUT_MS must be in (0, 10000]")
	}
	if cfg.IndexClaimTTL <= cfg.IndexTimeout+IndexLeaseSafetyMargin {
		return fmt.Errorf("INDEX_CLAIM_TTL_SECONDS must be greater than INDEX_TIMEOUT_MS plus 1s")
	}
	return nil
}

func validateIndexingConfig(cfg Config) error {
	if !cfg.AdminIndexingEnabled {
		return nil
	}
	if !cfg.AdminIngestionEnabled {
		return fmt.Errorf("ADMIN_INGESTION_ENABLED must be true when ADMIN_INDEXING_ENABLED=true")
	}
	if strings.TrimSpace(cfg.AIServiceURL) == "" || strings.TrimSpace(cfg.AIServiceToken) == "" {
		return fmt.Errorf("AI_SERVICE_URL and AI_SERVICE_TOKEN are required when ADMIN_INDEXING_ENABLED=true")
	}
	if len(strings.TrimSpace(cfg.AIServiceToken)) < 16 {
		return fmt.Errorf("AI_SERVICE_TOKEN must be at least 16 characters")
	}
	return nil
}

func validateIngestionConfig(cfg Config) error {
	if !cfg.AdminIngestionEnabled {
		return nil
	}
	if strings.TrimSpace(cfg.ObjectStorageEndpoint) == "" ||
		strings.TrimSpace(cfg.ObjectStorageAccessKey) == "" ||
		strings.TrimSpace(cfg.ObjectStorageSecretKey) == "" ||
		strings.TrimSpace(cfg.ObjectStorageBucket) == "" ||
		!cfg.ObjectStorageSSLConfigured {
		return fmt.Errorf("object storage settings are required when ADMIN_INGESTION_ENABLED=true")
	}
	if cfg.DocumentMaxBytes <= 0 || cfg.DocumentMaxBytes > DocumentMaxHardBytes {
		return fmt.Errorf("DOCUMENT_MAX_BYTES must be in (0, %d]", DocumentMaxHardBytes)
	}
	return nil
}

func validateAIConfig(cfg Config) error {
	if cfg.AIExtractTimeoutMS <= 0 || cfg.AIExtractTimeoutMS > AIExtractTimeoutHardMaxMS {
		return fmt.Errorf("AI_EXTRACT_TIMEOUT_MS must be in (0, %d]", AIExtractTimeoutHardMaxMS)
	}
	for _, item := range []struct {
		name string
		v    float64
	}{
		{"AI_INTENT_SELECT_MIN", cfg.AIIntentSelectMin},
		{"AI_INTENT_CONFIRM_MIN", cfg.AIIntentConfirmMin},
		{"AI_SLOT_CONFIDENCE_MIN", cfg.AISlotConfidenceMin},
	} {
		if math.IsNaN(item.v) || math.IsInf(item.v, 0) {
			return fmt.Errorf("%s must be a finite number", item.name)
		}
	}
	if cfg.AIIntentConfirmMin < 0 || cfg.AIIntentConfirmMin > 1 ||
		cfg.AIIntentSelectMin < 0 || cfg.AIIntentSelectMin > 1 ||
		cfg.AIIntentConfirmMin > cfg.AIIntentSelectMin {
		return fmt.Errorf("AI intent thresholds must satisfy 0 <= confirm_min <= select_min <= 1")
	}
	if cfg.AISlotConfidenceMin < 0 || cfg.AISlotConfidenceMin > 1 {
		return fmt.Errorf("AI_SLOT_CONFIDENCE_MIN must be in [0, 1]")
	}
	token := strings.TrimSpace(cfg.AIServiceToken)
	if token != "" && len(token) < 16 {
		return fmt.Errorf("AI_SERVICE_TOKEN must be at least 16 characters")
	}
	if cfg.AIExtractEnabled {
		if strings.TrimSpace(cfg.AIServiceURL) == "" {
			return fmt.Errorf("AI_SERVICE_URL is required when AI_EXTRACT_ENABLED=true")
		}
		if token == "" {
			return fmt.Errorf("AI_SERVICE_TOKEN is required when AI_EXTRACT_ENABLED=true")
		}
	}
	return nil
}

func applyEnvOverrides(cfg *Config) error {
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
	if v := os.Getenv("JWT_SECRET"); v != "" {
		cfg.JWTSecret = v
	}
	if v := os.Getenv("JWT_EXPIRE_HOURS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.JWTExpireHours = n
		}
	}
	if v := os.Getenv("CITIZEN_DOMAIN_IDS"); v != "" {
		parts := strings.Split(v, ",")
		ids := make([]string, 0, len(parts))
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if p != "" {
				ids = append(ids, p)
			}
		}
		if len(ids) > 0 {
			cfg.CitizenDomainIDs = ids
		}
	}
	if v, ok := os.LookupEnv("AI_EXTRACT_ENABLED"); ok && strings.TrimSpace(v) != "" {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "1", "true", "yes", "on":
			cfg.AIExtractEnabled = true
		case "0", "false", "no", "off":
			cfg.AIExtractEnabled = false
		default:
			return fmt.Errorf("AI_EXTRACT_ENABLED must be true or false")
		}
	}
	if v := os.Getenv("AI_SERVICE_URL"); v != "" {
		cfg.AIServiceURL = v
	}
	if v := os.Getenv("AI_SERVICE_TOKEN"); v != "" {
		cfg.AIServiceToken = v
	}
	if v, ok := os.LookupEnv("AI_EXTRACT_TIMEOUT_MS"); ok && strings.TrimSpace(v) != "" {
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return fmt.Errorf("AI_EXTRACT_TIMEOUT_MS must be an integer")
		}
		cfg.AIExtractTimeoutMS = n
	}
	if v, ok := os.LookupEnv("AI_INTENT_SELECT_MIN"); ok && strings.TrimSpace(v) != "" {
		f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
		if err != nil {
			return fmt.Errorf("AI_INTENT_SELECT_MIN must be a finite number")
		}
		cfg.AIIntentSelectMin = f
	}
	if v, ok := os.LookupEnv("AI_INTENT_CONFIRM_MIN"); ok && strings.TrimSpace(v) != "" {
		f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
		if err != nil {
			return fmt.Errorf("AI_INTENT_CONFIRM_MIN must be a finite number")
		}
		cfg.AIIntentConfirmMin = f
	}
	if v, ok := os.LookupEnv("AI_SLOT_CONFIDENCE_MIN"); ok && strings.TrimSpace(v) != "" {
		f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
		if err != nil {
			return fmt.Errorf("AI_SLOT_CONFIDENCE_MIN must be a finite number")
		}
		cfg.AISlotConfidenceMin = f
	}
	if v, ok := os.LookupEnv("ADMIN_INGESTION_ENABLED"); ok && strings.TrimSpace(v) != "" {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "1", "true", "yes", "on":
			cfg.AdminIngestionEnabled = true
		case "0", "false", "no", "off":
			cfg.AdminIngestionEnabled = false
		default:
			return fmt.Errorf("ADMIN_INGESTION_ENABLED must be true or false")
		}
	}
	if v := os.Getenv("OBJECT_STORAGE_ENDPOINT"); v != "" {
		cfg.ObjectStorageEndpoint = strings.TrimSpace(v)
	}
	if v := os.Getenv("OBJECT_STORAGE_ACCESS_KEY"); v != "" {
		cfg.ObjectStorageAccessKey = v
	}
	if v := os.Getenv("OBJECT_STORAGE_SECRET_KEY"); v != "" {
		cfg.ObjectStorageSecretKey = v
	}
	if v := os.Getenv("OBJECT_STORAGE_BUCKET"); v != "" {
		cfg.ObjectStorageBucket = strings.TrimSpace(v)
	}
	if v, ok := os.LookupEnv("OBJECT_STORAGE_USE_SSL"); ok && strings.TrimSpace(v) != "" {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "1", "true", "yes", "on":
			cfg.ObjectStorageUseSSL = true
			cfg.ObjectStorageSSLConfigured = true
		case "0", "false", "no", "off":
			cfg.ObjectStorageUseSSL = false
			cfg.ObjectStorageSSLConfigured = true
		default:
			return fmt.Errorf("OBJECT_STORAGE_USE_SSL must be true or false")
		}
	}
	if v, ok := os.LookupEnv("DOCUMENT_MAX_BYTES"); ok && strings.TrimSpace(v) != "" {
		n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
		if err != nil {
			return fmt.Errorf("DOCUMENT_MAX_BYTES must be an integer")
		}
		cfg.DocumentMaxBytes = n
	}
	if v, ok := os.LookupEnv("ADMIN_INDEXING_ENABLED"); ok && strings.TrimSpace(v) != "" {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "1", "true", "yes", "on":
			cfg.AdminIndexingEnabled = true
		case "0", "false", "no", "off":
			cfg.AdminIndexingEnabled = false
		default:
			return fmt.Errorf("ADMIN_INDEXING_ENABLED must be true or false")
		}
	}
	if v, ok := os.LookupEnv("INDEX_MODE"); ok && strings.TrimSpace(v) != "" {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "mock", "pipeline":
			cfg.IndexMode = strings.ToLower(strings.TrimSpace(v))
		default:
			return fmt.Errorf("INDEX_MODE must be mock or pipeline")
		}
	}
	if v, ok := os.LookupEnv("INDEX_CLAIM_TTL_SECONDS"); ok && strings.TrimSpace(v) != "" {
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return fmt.Errorf("INDEX_CLAIM_TTL_SECONDS must be an integer")
		}
		cfg.IndexClaimTTL = time.Duration(n) * time.Second
	}
	if v, ok := os.LookupEnv("INDEX_TIMEOUT_MS"); ok && strings.TrimSpace(v) != "" {
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return fmt.Errorf("INDEX_TIMEOUT_MS must be an integer")
		}
		cfg.IndexTimeout = time.Duration(n) * time.Millisecond
	}
	if v, ok := os.LookupEnv("INDEX_PIPELINE_TIMEOUT_SECONDS"); ok && strings.TrimSpace(v) != "" {
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return fmt.Errorf("INDEX_PIPELINE_TIMEOUT_SECONDS must be an integer")
		}
		cfg.IndexPipelineTimeout = time.Duration(n) * time.Second
	}
	if v, ok := os.LookupEnv("INDEX_PUBLISH_TIMEOUT_MS"); ok && strings.TrimSpace(v) != "" {
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return fmt.Errorf("INDEX_PUBLISH_TIMEOUT_MS must be an integer")
		}
		cfg.IndexPublishTimeout = time.Duration(n) * time.Millisecond
	}
	return nil
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
