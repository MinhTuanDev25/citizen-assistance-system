package activation

import (
	"fmt"
	"net"
	"net/url"
	"os"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	P4TestDatabase    = "cas_p4_test"
	p4TestDefaultPort = 5432
)

var p4LibpqOverrideEnv = []string{
	"PGHOST", "PGHOSTADDR", "PGPORT", "PGDATABASE", "PGSERVICE", "PGSERVICEFILE", "PGSYSCONFDIR",
}

var p4AllowedHosts = map[string]struct{}{
	"127.0.0.1": {},
	"localhost": {},
}

var p4AllowedQuery = map[string]struct{}{
	"sslmode": {},
}

// IsolationError is a fail-closed Phase 4 integration DSN rejection.
type IsolationError struct {
	Reason string
}

func (e IsolationError) Error() string { return e.Reason }

// ValidateP4TestDSN checks the DSN the way pgx will parse it. It does not connect.
func ValidateP4TestDSN(raw string, getenv func(string) string) error {
	if getenv == nil {
		getenv = os.Getenv
	}
	if strings.TrimSpace(getenv("CAS_P4_TEST")) != "1" && strings.TrimSpace(getenv("CAS_P4_TEST")) != "true" {
		return IsolationError{Reason: "CAS_P4_TEST=1 required"}
	}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return IsolationError{Reason: "CAS_P4_TEST_DATABASE_URL required"}
	}
	if present := libpqOverrides(getenv); len(present) > 0 {
		return IsolationError{Reason: "refusing libpq environment overrides: " + strings.Join(present, ",")}
	}
	if !strings.HasPrefix(raw, "postgres://") && !strings.HasPrefix(raw, "postgresql://") {
		return IsolationError{Reason: "CAS_P4_TEST_DATABASE_URL must be a postgres URL"}
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return IsolationError{Reason: "malformed DSN"}
	}
	if strings.Contains(parsed.Host, ",") || strings.Contains(parsed.Hostname(), ",") {
		return IsolationError{Reason: "refusing a multi-host or socket PostgreSQL target"}
	}
	keys := queryKeys(parsed.RawQuery)
	if len(keys) != uniqueCount(keys) {
		return IsolationError{Reason: "duplicate PostgreSQL connection parameters"}
	}
	for _, key := range keys {
		if _, ok := p4AllowedQuery[key]; !ok {
			return IsolationError{Reason: "PostgreSQL query parameter can change the connection target"}
		}
	}
	cfg, err := pgxpool.ParseConfig(raw)
	if err != nil {
		return IsolationError{Reason: "PostgreSQL DSN is not accepted"}
	}
	host := strings.TrimSpace(cfg.ConnConfig.Host)
	if host == "" || strings.Contains(host, ",") || strings.HasPrefix(host, "/") || strings.HasPrefix(host, "@") {
		return IsolationError{Reason: "refusing a multi-host or socket PostgreSQL target"}
	}
	if ip := net.ParseIP(host); ip != nil && ip.IsUnspecified() {
		return IsolationError{Reason: "refusing an unspecified PostgreSQL host"}
	}
	if _, ok := p4AllowedHosts[host]; !ok {
		return IsolationError{Reason: "refusing a Postgres endpoint outside the dedicated test hosts"}
	}
	port := int(cfg.ConnConfig.Port)
	if port == 0 {
		port = p4TestDefaultPort
	}
	wantPort := p4TestDefaultPort
	if extra := strings.TrimSpace(getenv("CAS_P4_TEST_PORT")); extra != "" {
		n, err := strconv.Atoi(extra)
		if err != nil || n <= 0 {
			return IsolationError{Reason: "CAS_P4_TEST_PORT is invalid"}
		}
		wantPort = n
	}
	if port != wantPort {
		return IsolationError{Reason: fmt.Sprintf("refusing Postgres port %d; want %d", port, wantPort)}
	}
	dbname := strings.TrimSpace(cfg.ConnConfig.Database)
	if dbname != P4TestDatabase {
		return IsolationError{Reason: "refusing a database other than cas_p4_test"}
	}
	if strings.TrimSpace(cfg.ConnConfig.User) == "" || cfg.ConnConfig.Password == "" {
		return IsolationError{Reason: "test database credentials are required"}
	}
	ssl := ""
	if cfg.ConnConfig.TLSConfig == nil {
		ssl = "disable"
	}
	if q := parsed.Query().Get("sslmode"); q != "" {
		ssl = q
	}
	if ssl != "disable" {
		return IsolationError{Reason: "the test database connection must disable TLS"}
	}
	return nil
}

func libpqOverrides(getenv func(string) string) []string {
	var out []string
	for _, name := range p4LibpqOverrideEnv {
		if strings.TrimSpace(getenv(name)) != "" {
			out = append(out, name)
		}
	}
	return out
}

func queryKeys(rawQuery string) []string {
	if rawQuery == "" {
		return nil
	}
	out := make([]string, 0)
	for _, part := range strings.Split(rawQuery, "&") {
		if part == "" {
			continue
		}
		key, _, _ := strings.Cut(part, "=")
		key, _ = url.QueryUnescape(key)
		out = append(out, strings.ToLower(strings.TrimSpace(key)))
	}
	return out
}

func uniqueCount(in []string) int {
	seen := map[string]struct{}{}
	for _, s := range in {
		seen[s] = struct{}{}
	}
	return len(seen)
}
