// Command seed-demo inserts local-only demo logins.
// Guard: refuses to run unless APP_ENV=local (or CAS_ALLOW_DEMO_SEED=1 for explicit override).
package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/config"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/db"
	"github.com/google/uuid"
)

func main() {
	env := strings.ToLower(strings.TrimSpace(os.Getenv("APP_ENV")))
	allow := os.Getenv("CAS_ALLOW_DEMO_SEED") == "1"
	if env != "local" && !allow {
		fmt.Fprintln(os.Stderr, "seed-demo refused: APP_ENV must be local (or set CAS_ALLOW_DEMO_SEED=1)")
		os.Exit(2)
	}

	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	pool, err := db.NewPool(ctx, cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer pool.Close()

	adminHash, err := auth.HashPassword(envOr("CAS_DEMO_ADMIN_PASSWORD", "admin123"))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	citizenHash, err := auth.HashPassword(envOr("CAS_DEMO_CITIZEN_PASSWORD", "citizen123"))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	adminID := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	citizenID := uuid.MustParse("bbbbbbbb-cccc-dddd-eeee-ffffffffffff")

	_, err = pool.Exec(ctx, `
		INSERT INTO users (id, role, full_name, email, password_hash)
		VALUES ($1, 'ADMIN', 'Cán bộ One Cửa (local demo)', 'admin@chuse.vn', $2)
		ON CONFLICT (id) DO UPDATE
		SET role='ADMIN', full_name=EXCLUDED.full_name, email=EXCLUDED.email,
		    password_hash=EXCLUDED.password_hash, updated_at=now()`,
		adminID, adminHash)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	_, err = pool.Exec(ctx, `
		INSERT INTO users (id, role, full_name, email, password_hash)
		VALUES ($1, 'CITIZEN', 'Công dân demo (local)', 'citizen@example.com', $2)
		ON CONFLICT (id) DO UPDATE
		SET role='CITIZEN', full_name=EXCLUDED.full_name, email=EXCLUDED.email,
		    password_hash=EXCLUDED.password_hash, updated_at=now()`,
		citizenID, citizenHash)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("seed-demo OK (local only): admin@chuse.vn + citizen@example.com")
}

func envOr(k, fallback string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return fallback
}
