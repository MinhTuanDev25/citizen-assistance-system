package config

import (
	"fmt"
	"math"
	"net"
	"net/url"
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
	AIRetrieveEnabled   bool
	AIRetrieveTimeoutMS int
	AIDraftTimeoutMS    int

	// Admin PDF intake (P3.1). Off by default: the API does not open object storage.
	AdminIngestionEnabled bool
	AdminIndexingEnabled  bool
	IndexMode             string
	IndexClaimTTL         time.Duration
	IndexTimeout          time.Duration
	IndexPipelineTimeout  time.Duration
	IndexPublishTimeout   time.Duration
	// IndexAsync runs pipeline indexing as a background job: the admin request
	// returns once the job is claimed, and a heartbeat renews the lease.
	IndexAsync                    bool
	IndexHeartbeat                time.Duration
	IndexMaxConcurrent            int
	ObjectStorageEndpoint         string
	ObjectStorageAccessKey        string
	ObjectStorageSecretKey        string
	ObjectStorageBucket           string
	ObjectStorageUseSSL           bool
	ObjectStorageSSLConfigured    bool
	ObjectStorageRegion           string
	ObjectStorageAutoCreateBucket bool
	DocumentMaxBytes              int64

	// Speech translation (Step 7.5). Off by default: text chat is unchanged.
	SpeechEnabled        bool
	SpeechServiceURL     string
	SpeechServiceToken   string
	SpeechTimeoutMS      int
	SpeechMaxUploadBytes int64
	SpeechMaxInflight    int
}

// DocumentMaxHardBytes matches middleware.DocumentUploadHardMax.
const DocumentMaxHardBytes int64 = 64 << 20

// AIExtractTimeoutHardMaxMS matches aiextract.HardMaxTimeout. A configured
// timeout above this is a startup error, not a silent clamp.
const AIExtractTimeoutHardMaxMS = 8000

// AIRetrieveTimeoutHardMaxMS is the retrieve ceiling. Local ONNX cold start
// can exceed 3s; 10000 is a test value, not a production default.
const AIRetrieveTimeoutHardMaxMS = 10000

// AIDraftTimeoutHardMaxMS is the Admin drafting ceiling. This is independent
// of the citizen extract 8s cap.
const AIDraftTimeoutHardMaxMS = 310000

// SpeechTimeoutHardMaxMS is above the upstream SPEECH_REQUEST_TIMEOUT_S (120s)
// so a legitimate GPU call is not cut by the Go client first.
const SpeechTimeoutHardMaxMS = 180000

// SpeechUploadHardMaxBytes matches the upstream default 10 MiB HTTP body cap.
const SpeechUploadHardMaxBytes int64 = 10 << 20

// SpeechUploadMinBytes is the smallest SPEECH_MAX_UPLOAD_BYTES that still
// leaves room for a WAV after the 64 KiB multipart reserve. Equal to
// speech.MultipartOverheadReserve + 1. A cap at or below the reserve would
// make MaxAudioBytes return 0; that must fail validation, not fail-open.
const SpeechUploadMinBytes int64 = 64<<10 + 1

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

		AIExtractEnabled:              false,
		AIServiceURL:                  "",
		AIExtractTimeoutMS:            3000,
		AIIntentSelectMin:             0.82,
		AIIntentConfirmMin:            0.55,
		AISlotConfidenceMin:           0.6,
		AIRetrieveEnabled:             false,
		AIRetrieveTimeoutMS:           3000,
		AIDraftTimeoutMS:              45000,
		IndexMode:                     "mock",
		IndexClaimTTL:                 30 * time.Second,
		IndexTimeout:                  3 * time.Second,
		IndexPipelineTimeout:          2 * time.Minute,
		IndexPublishTimeout:           10 * time.Second,
		IndexHeartbeat:                15 * time.Second,
		IndexMaxConcurrent:            1,
		ObjectStorageAutoCreateBucket: true,
		SpeechEnabled:                 false,
		SpeechTimeoutMS:               130000,
		SpeechMaxUploadBytes:          SpeechUploadHardMaxBytes,
		SpeechMaxInflight:             1,
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
	if err := validateSpeechConfig(cfg); err != nil {
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
	if err := validateProduction(cfg); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

// IndexLeaseSafetyMargin keeps a crashed claim recoverable after the worker timeout.
const IndexLeaseSafetyMargin = time.Second

// IndexAsyncMaxPipelineTimeout bounds one background job, matching the AI service ceiling.
const IndexAsyncMaxPipelineTimeout = 2 * time.Hour

func validateIndexLease(cfg Config) error {
	if cfg.IndexMode == "pipeline" && cfg.IndexAsync {
		// The heartbeat renews the lease, so the TTL only has to outlive a few
		// missed beats; a crashed API frees the link for Retry within one TTL.
		if cfg.IndexPipelineTimeout <= 0 || cfg.IndexPipelineTimeout > IndexAsyncMaxPipelineTimeout {
			return fmt.Errorf("INDEX_PIPELINE_TIMEOUT_SECONDS must be in (0, 7200] when INDEX_ASYNC=true")
		}
		if cfg.IndexHeartbeat < time.Second || cfg.IndexHeartbeat > 2*time.Minute {
			return fmt.Errorf("INDEX_HEARTBEAT_SECONDS must be in [1, 120]")
		}
		if cfg.IndexClaimTTL < 3*cfg.IndexHeartbeat || cfg.IndexClaimTTL > 16*time.Minute {
			return fmt.Errorf("INDEX_CLAIM_TTL_SECONDS must be at least 3x INDEX_HEARTBEAT_SECONDS and at most 960 when INDEX_ASYNC=true")
		}
		if cfg.IndexMaxConcurrent < 1 || cfg.IndexMaxConcurrent > 4 {
			return fmt.Errorf("INDEX_MAX_CONCURRENT must be in [1, 4]")
		}
		if cfg.IndexPublishTimeout <= 0 || cfg.IndexPublishTimeout > time.Minute {
			return fmt.Errorf("INDEX_PUBLISH_TIMEOUT_MS must be in (0, 60000]")
		}
		return nil
	}
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

func validateProduction(cfg Config) error {
	if cfg.Env != "prod" {
		return nil
	}
	if isDevPlaceholder(cfg.JWTSecret) {
		return fmt.Errorf("JWT_SECRET is a development placeholder")
	}
	if !productionDatabaseTLS(cfg.DatabaseURL) {
		return fmt.Errorf("DATABASE_URL in APP_ENV=prod must set sslmode=require, verify-ca, or verify-full")
	}
	if isDevPlaceholder(cfg.AIServiceToken) {
		return fmt.Errorf("AI_SERVICE_TOKEN is a development placeholder")
	}
	if cfg.SpeechEnabled && isDevPlaceholder(cfg.SpeechServiceToken) {
		return fmt.Errorf("SPEECH_SERVICE_TOKEN is a development placeholder")
	}
	if cfg.AdminIngestionEnabled {
		if cfg.ObjectStorageAutoCreateBucket {
			return fmt.Errorf("OBJECT_STORAGE_AUTO_CREATE_BUCKET must be false when APP_ENV=prod")
		}
		if cfg.ObjectStorageAccessKey == "minioadmin" || cfg.ObjectStorageSecretKey == "minioadmin" {
			return fmt.Errorf("object storage credentials are development defaults")
		}
	}
	if cfg.AdminIndexingEnabled && cfg.IndexMode != "pipeline" {
		return fmt.Errorf("INDEX_MODE=pipeline is required when APP_ENV=prod and indexing is enabled")
	}
	return nil
}

func isDevPlaceholder(value string) bool {
	text := strings.ToLower(strings.TrimSpace(value))
	if text == "" {
		return false
	}
	return strings.Contains(text, "local-dev") || strings.Contains(text, "change-me") || strings.Contains(text, "minioadmin")
}

func productionDatabaseTLS(dsn string) bool {
	parsed, err := url.Parse(dsn)
	if err != nil {
		return false
	}
	mode := strings.ToLower(parsed.Query().Get("sslmode"))
	return mode == "require" || mode == "verify-ca" || mode == "verify-full"
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

func isRunPodEndpoint(endpoint string) bool {
	return strings.Contains(strings.ToLower(endpoint), "runpod")
}

func runPodPortError(endpoint string) error {
	text := strings.TrimSpace(endpoint)
	port := ""
	if strings.Contains(text, "://") {
		parsed, err := url.Parse(text)
		if err != nil || parsed.Hostname() == "" {
			return fmt.Errorf("object storage endpoint is invalid")
		}
		port = parsed.Port()
	} else if _, parsedPort, err := net.SplitHostPort(text); err == nil {
		port = parsedPort
	}
	if port != "" && port != "443" {
		return fmt.Errorf("RunPod object storage requires port 443")
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
	if isRunPodEndpoint(cfg.ObjectStorageEndpoint) {
		lower := strings.ToLower(strings.TrimSpace(cfg.ObjectStorageEndpoint))
		if strings.HasPrefix(lower, "http://") {
			return fmt.Errorf("RunPod object storage requires TLS")
		}
		if !strings.HasPrefix(lower, "https://") && !cfg.ObjectStorageUseSSL {
			return fmt.Errorf("OBJECT_STORAGE_USE_SSL must be true for RunPod object storage")
		}
		if err := runPodPortError(cfg.ObjectStorageEndpoint); err != nil {
			return err
		}
		if strings.TrimSpace(cfg.ObjectStorageRegion) == "" {
			return fmt.Errorf("OBJECT_STORAGE_REGION is required for RunPod object storage")
		}
		if cfg.ObjectStorageAutoCreateBucket {
			return fmt.Errorf("OBJECT_STORAGE_AUTO_CREATE_BUCKET must be false for RunPod object storage")
		}
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
	if cfg.AIDraftTimeoutMS <= 0 || cfg.AIDraftTimeoutMS > AIDraftTimeoutHardMaxMS {
		return fmt.Errorf("AI_DRAFT_TIMEOUT_MS must be in (0, %d]", AIDraftTimeoutHardMaxMS)
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
	if cfg.AIRetrieveTimeoutMS <= 0 || cfg.AIRetrieveTimeoutMS > AIRetrieveTimeoutHardMaxMS {
		return fmt.Errorf("AI_RETRIEVE_TIMEOUT_MS must be in (0, %d]", AIRetrieveTimeoutHardMaxMS)
	}
	if cfg.AIRetrieveEnabled {
		if strings.TrimSpace(cfg.AIServiceURL) == "" {
			return fmt.Errorf("AI_SERVICE_URL is required when AI_RETRIEVE_ENABLED=true")
		}
		if token == "" {
			return fmt.Errorf("AI_SERVICE_TOKEN is required when AI_RETRIEVE_ENABLED=true")
		}
	}
	return nil
}

func validateSpeechConfig(cfg Config) error {
	if cfg.SpeechTimeoutMS <= 0 || cfg.SpeechTimeoutMS > SpeechTimeoutHardMaxMS {
		return fmt.Errorf("SPEECH_TIMEOUT_MS must be in (0, %d]", SpeechTimeoutHardMaxMS)
	}
	if cfg.SpeechMaxUploadBytes < SpeechUploadMinBytes || cfg.SpeechMaxUploadBytes > SpeechUploadHardMaxBytes {
		return fmt.Errorf("SPEECH_MAX_UPLOAD_BYTES must be in [%d, %d]", SpeechUploadMinBytes, SpeechUploadHardMaxBytes)
	}
	if cfg.SpeechMaxInflight < 1 || cfg.SpeechMaxInflight > 4 {
		return fmt.Errorf("SPEECH_MAX_INFLIGHT must be in [1, 4]")
	}
	token := strings.TrimSpace(cfg.SpeechServiceToken)
	if token != "" && len(token) < 16 {
		return fmt.Errorf("SPEECH_SERVICE_TOKEN must be at least 16 characters")
	}
	if !cfg.SpeechEnabled {
		return nil
	}
	if strings.TrimSpace(cfg.SpeechServiceURL) == "" {
		return fmt.Errorf("SPEECH_SERVICE_URL is required when SPEECH_ENABLED=true")
	}
	if token == "" {
		return fmt.Errorf("SPEECH_SERVICE_TOKEN is required when SPEECH_ENABLED=true")
	}
	if err := validateSpeechServiceURL(cfg.SpeechServiceURL); err != nil {
		return err
	}
	return nil
}

func validateSpeechServiceURL(raw string) error {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return fmt.Errorf("SPEECH_SERVICE_URL is invalid")
	}
	scheme := strings.ToLower(parsed.Scheme)
	if scheme != "http" {
		return fmt.Errorf("SPEECH_SERVICE_URL scheme must be http to a loopback address")
	}
	if parsed.User != nil {
		return fmt.Errorf("SPEECH_SERVICE_URL must not include credentials")
	}
	if parsed.RawQuery != "" {
		return fmt.Errorf("SPEECH_SERVICE_URL must not include a query")
	}
	if parsed.Fragment != "" {
		return fmt.Errorf("SPEECH_SERVICE_URL must not include a fragment")
	}
	path := parsed.EscapedPath()
	if path != "" && path != "/" {
		return fmt.Errorf("SPEECH_SERVICE_URL must not include a path")
	}
	if parsed.Host == "" || parsed.Hostname() == "" {
		return fmt.Errorf("SPEECH_SERVICE_URL is invalid")
	}
	if !speechLoopbackHost(parsed.Hostname()) {
		return fmt.Errorf("SPEECH_SERVICE_URL host must be loopback (127.0.0.1, ::1, or localhost)")
	}
	return nil
}

func speechLoopbackHost(host string) bool {
	name := strings.ToLower(strings.TrimSpace(host))
	if name == "localhost" {
		return true
	}
	ip := net.ParseIP(name)
	return ip != nil && ip.IsLoopback()
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
	if v, ok := os.LookupEnv("AI_RETRIEVE_ENABLED"); ok && strings.TrimSpace(v) != "" {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "1", "true", "yes", "on":
			cfg.AIRetrieveEnabled = true
		case "0", "false", "no", "off":
			cfg.AIRetrieveEnabled = false
		default:
			return fmt.Errorf("AI_RETRIEVE_ENABLED must be true or false")
		}
	}
	if v, ok := os.LookupEnv("AI_RETRIEVE_TIMEOUT_MS"); ok && strings.TrimSpace(v) != "" {
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return fmt.Errorf("AI_RETRIEVE_TIMEOUT_MS must be an integer")
		}
		cfg.AIRetrieveTimeoutMS = n
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
	if v, ok := os.LookupEnv("AI_DRAFT_TIMEOUT_MS"); ok && strings.TrimSpace(v) != "" {
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return fmt.Errorf("AI_DRAFT_TIMEOUT_MS must be an integer")
		}
		cfg.AIDraftTimeoutMS = n
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
	if v := os.Getenv("OBJECT_STORAGE_REGION"); v != "" {
		cfg.ObjectStorageRegion = strings.TrimSpace(v)
	}
	if v, ok := os.LookupEnv("OBJECT_STORAGE_AUTO_CREATE_BUCKET"); ok && strings.TrimSpace(v) != "" {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "1", "true", "yes", "on":
			cfg.ObjectStorageAutoCreateBucket = true
		case "0", "false", "no", "off":
			cfg.ObjectStorageAutoCreateBucket = false
		default:
			return fmt.Errorf("OBJECT_STORAGE_AUTO_CREATE_BUCKET must be true or false")
		}
	} else if isRunPodEndpoint(cfg.ObjectStorageEndpoint) {
		cfg.ObjectStorageAutoCreateBucket = false
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
	if v, ok := os.LookupEnv("INDEX_ASYNC"); ok && strings.TrimSpace(v) != "" {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "1", "true", "yes", "on":
			cfg.IndexAsync = true
		case "0", "false", "no", "off":
			cfg.IndexAsync = false
		default:
			return fmt.Errorf("INDEX_ASYNC must be true or false")
		}
	}
	if v, ok := os.LookupEnv("INDEX_HEARTBEAT_SECONDS"); ok && strings.TrimSpace(v) != "" {
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return fmt.Errorf("INDEX_HEARTBEAT_SECONDS must be an integer")
		}
		cfg.IndexHeartbeat = time.Duration(n) * time.Second
	}
	if v, ok := os.LookupEnv("INDEX_MAX_CONCURRENT"); ok && strings.TrimSpace(v) != "" {
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return fmt.Errorf("INDEX_MAX_CONCURRENT must be an integer")
		}
		cfg.IndexMaxConcurrent = n
	}
	if v, ok := os.LookupEnv("SPEECH_ENABLED"); ok && strings.TrimSpace(v) != "" {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "1", "true", "yes", "on":
			cfg.SpeechEnabled = true
		case "0", "false", "no", "off":
			cfg.SpeechEnabled = false
		default:
			return fmt.Errorf("SPEECH_ENABLED must be true or false")
		}
	}
	if v := os.Getenv("SPEECH_SERVICE_URL"); v != "" {
		cfg.SpeechServiceURL = strings.TrimSpace(v)
	}
	if v := os.Getenv("SPEECH_SERVICE_TOKEN"); v != "" {
		cfg.SpeechServiceToken = v
	}
	if v, ok := os.LookupEnv("SPEECH_TIMEOUT_MS"); ok && strings.TrimSpace(v) != "" {
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return fmt.Errorf("SPEECH_TIMEOUT_MS must be an integer")
		}
		cfg.SpeechTimeoutMS = n
	}
	if v, ok := os.LookupEnv("SPEECH_MAX_UPLOAD_BYTES"); ok && strings.TrimSpace(v) != "" {
		n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
		if err != nil {
			return fmt.Errorf("SPEECH_MAX_UPLOAD_BYTES must be an integer")
		}
		cfg.SpeechMaxUploadBytes = n
	}
	if v, ok := os.LookupEnv("SPEECH_MAX_INFLIGHT"); ok && strings.TrimSpace(v) != "" {
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return fmt.Errorf("SPEECH_MAX_INFLIGHT must be an integer")
		}
		cfg.SpeechMaxInflight = n
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
